from __future__ import annotations
from typing import Optional, Union, Literal, List
from dataclasses import dataclass, field
from copy import deepcopy

import torch
from torch import Tensor
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertLayer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import get_scheduler
from transformers import AdamW
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.integrations import TensorBoardCallback
from datasets import load_dataset



MODEL_NAME = "google-bert/bert-base-uncased"
EXPERIMENT_NAME = "main-experiment"

CONTINUE_FROM = None
OUTPUT_DIR = 'bert-simple-main-experiment-gradual-patch-no-clone'


ModelSource = Optional[Union[Literal['local'], Literal['hf']]]
DataSource = Union[Literal['bookcorpus'], Literal['wikipedia']]


def load_model_from(source: ModelSource = None):
    if source is None:
        config = AutoConfig.from_pretrained(MODEL_NAME)
        model = AutoModelForMaskedLM.from_config(config)
    elif source == 'local':
        model = AutoModelForMaskedLM.from_pretrained(CONTINUE_FROM)
        # model = AutoModelForMaskedLM.from_pretrained(OUTPUT_DIR)
    elif source == 'hf':
        model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    else:
        raise ValueError(f'model source {source} is not supported')
    assert isinstance(model, BertForMaskedLM)
    return model


def load_data_from(source: DataSource, dataset_slice=None):
    # https://huggingface.co/datasets/wikimedia/wikipedia
    if source == 'bookcorpus':
        # return load_dataset("wikimedia/wikipedia", "20231101.en")
        return load_dataset('bookcorpus', split=dataset_slice)
    elif source == 'wikipedia':
        raise ValueError('TODO')
    raise ValueError(f'data source {source} is not supported')


# https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling
def group_texts(block_size, item):
    # Concatenate all texts.
    concatenated_examples = {k: sum(item[k], []) for k in item.keys()}
    total_length = len(concatenated_examples[list(item.keys())[0]])

    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # Split by chunks of block_size.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_optimizer(optimizer):
    return sum(p.numel() for group in optimizer.param_groups for p in group['params'])


# https://discuss.huggingface.co/t/logging-text-using-model-outputs-with-tensorboard/46621/5
class ParameterCountCallback(TensorBoardCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        optimizer = kwargs['optimizer']

        # for safety, check if the tensorboard writer is initialized
        if self.tb_writer is None:
            self._init_summary_writer(args)

        self.tb_writer.add_scalar(
            tag="num_encoder_layers",
            scalar_value=len(model.bert.encoder.layer),
            global_step=state.global_step
        )

        self.tb_writer.add_scalar(
            tag="num_encoder_parameters",
            scalar_value=count_parameters(model.bert.encoder.layer),
            global_step=state.global_step
        )

        self.tb_writer.add_scalar(
            tag="num_optimizer_parameters",
            scalar_value=count_parameters_optimizer(optimizer),
            global_step=state.global_step
        )


@dataclass
class Patch:
    training_progress: float
    insert_at: int
    clone_from: Optional[int] = None


class PatchingCallback(TrainerCallback):
    def __init__(self, patches: list[Optional[Patch]], total_training_steps: float):
        self.patches = sorted(patches, key=lambda patch: patch.training_progress)
        self.patch_i = 0
        self.total_training_steps = total_training_steps

    # [available arguments](https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/callback#transformers.TrainerCallback)
    # Return type is [`TrainerControl`](https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/callback#transformers.TrainerControl)
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        with torch.no_grad():
            while (
                self.patch_i < len(self.patches) and
                state.global_step > self.patches[self.patch_i].training_progress * self.total_training_steps
            ):
                model = kwargs['model']
                optimizer = kwargs['optimizer']
                config = model.bert.encoder.config

                patch = self.patches[self.patch_i]

                if patch is not None:
                    layer = list(model.bert.encoder.layer)

                    addition = (
                        deepcopy(layer[patch.clone_from]) if patch.clone_from is not None else BertLayer(config)
                    ).to(model.device)

                    layer_list = list(layer)
                    layer_list.insert(patch.insert_at, addition)
                    layer = nn.ModuleList(layer_list)

                    optimizer.add_param_group({"params": addition.parameters()})

                    model.bert.encoder.layer = nn.ModuleList(layer)
                    config.num_hidden_layers = len(layer)

                self.patch_i += 1


@dataclass
class HyperParameters:
    model_source: ModelSource = None
    data_source: DataSource = 'bookcorpus'
    model_initial_num_layers: Optional[int] = None
    model_patches: List[Patch] = field(default_factory=list)
    block_size: int = 128
    num_proc: int = 50
    mlm_probability: float = 0.1
    dataset_fraction: Optional[float] = None
    test_size: float = 0.05
    num_train_epochs: int = 1
    lr: float = 2e-5
    lr_end: float = 2e-5 * 2 / 3
    # examples: https://huggingface.co/docs/datasets/v1.11.0/splits.html#examples
    dataset_slice: Optional[str] = None
    per_device_train_batch_size: int = 32
    warmup_ratio: float = 0.025
    use_tensorboard = True


class Experiment:
    def __init__(self, hyperparameters: Optional[HyperParameters] = None):
        if hyperparameters is None:
            self.hyperparameters = HyperParameters()
        else:
            self.hyperparameters = hyperparameters

    def load_tokenizer(self):
        # https://huggingface.co/google-bert/bert-base-uncased?library=transformers
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        assert isinstance(self.tokenizer, BertTokenizerFast)

    def load_model(self):
        self.model = load_model_from(self.hyperparameters.model_source)

    def load_data(self):
        h = self.hyperparameters
        assert self.tokenizer is not None

        data = load_data_from(h.data_source)
        print(data)
        data = data['train'] # this dataset only has a `train` dataset.
        data = data.map(
            lambda item: self.tokenizer(item['text']),
            batched=True,
            num_proc=h.num_proc,
            remove_columns=data.column_names,
        )

        if h.dataset_fraction is not None:
            new_size = int(data.num_rows * h.dataset_fraction)
            data = data.select(range(new_size))

        data = data.train_test_split(test_size=h.test_size)

        data = data.map(
            lambda item: group_texts(h.block_size, item),
            batched=True,
            num_proc=h.num_proc,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=h.mlm_probability,
        )

        self.data = {
            'data': data,
            'data_collator': data_collator,
        }

    def patch_model(self):
        assert self.model is not None
        h = self.hyperparameters

        config = self.model.bert.encoder.config

        if h.model_initial_num_layers is not None:
            layer = [BertLayer(config) for _ in range(h.model_initial_num_layers)]
            self.model.bert.encoder.layer = nn.ModuleList(layer)

    def total_training_steps(self):
        num_training_examples = len(self.data['data']['train'])
        per_device_train_batch_size = self.hyperparameters.per_device_train_batch_size
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_train_epochs = self.hyperparameters.num_train_epochs

        return (num_training_examples // (per_device_train_batch_size * num_devices)) * num_train_epochs

    def custom_optimizer(self):
        h = self.hyperparameters

        optimizer = AdamW(
            self.model.parameters(),
            lr=h.lr
        )

        num_training_steps = self.total_training_steps()
        num_warmup_steps = int(num_training_steps * h.warmup_ratio)
        scheduler = get_scheduler(
            "polynomial",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs={
                'lr_end': h.lr_end,
                'power': 1,
            }
        )

        return optimizer, scheduler

    def build_training_arguments(self):
        h = self.hyperparameters

        return TrainingArguments(
            output_dir="bert-pretraining",
            # eval_strategy="epoch",
            learning_rate=h.lr,
            num_train_epochs=h.num_train_epochs,
            weight_decay=0.01,
            per_device_train_batch_size=h.per_device_train_batch_size,
            warmup_ratio=h.warmup_ratio,
            # dataloader_num_workers=num_proc,
            # dataloader_prefetch_factor=4,
            # limit the amount of evaluation results held on the GPU
            # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.eval_accumulation_steps
            # eval_accumulation_steps=1,
            # speed up evaluation
            # per_device_eval_batch_size=64,
            # logging & export
            push_to_hub=False,
            save_strategy='epoch',
            report_to='tensorboard' if h.use_tensorboard else 'none',
            logging_strategy='steps',
            logging_steps=500,
        )

    def build_trainer(self):
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.data is not None
        h = self.hyperparameters

        optimizer, scheduler = self.custom_optimizer()

        trainer = Trainer(
            model=self.model,
            args=self.build_training_arguments(),
            train_dataset=self.data['data']['train'],
            eval_dataset=self.data['data']['test'],
            data_collator=self.data['data_collator'],
            tokenizer=self.tokenizer,
            # compute_metrics=compute_metrics,
            optimizers=(optimizer, scheduler),
        )

        if h.model_patches is not None:
            patch_callback = PatchingCallback(patches=h.model_patches, total_training_steps=self.total_training_steps())
            trainer.add_callback(patch_callback)

            parameter_callback = ParameterCountCallback()
            trainer.add_callback(parameter_callback)

        return trainer

    def train(self):
        trainer = self.build_trainer()
        trainer.train()
        self.traienr = trainer

    def fill_masks(self, text: str):
        assert self.tokenizer is not None
        assert self.model is not None

        mask_token_id = self.tokenizer.mask_token_id
        tokenized = self.tokenizer(text)
        input_ids = tokenized['input_ids']
        model_out = self.model(
            input_ids=Tensor([tokenized['input_ids']]).long().to(self.model.device),
            attention_mask=Tensor([tokenized['attention_mask']]).long().to(self.model.device),
        )
        out_ids = model_out.logits.argmax(-1)[0].tolist()
        out_mask_ids = [out_ids[i] for i in range(len(out_ids)) if input_ids[i] == mask_token_id]
        return ', '.join([self.tokenizer.decode(token_id) for token_id in out_mask_ids])


if __name__ == '__main__':
    hyperparameters = HyperParameters()

    # hyperparameters.model_source = 'local'
    # hyperparameters.dataset_fraction = 0.1

    hyperparameters.model_initial_num_layers = 1
    hyperparameters.model_patches = [
        # initial state: a                        tota: 1
        Patch(0.1, 1), # a b                      tota: 2
        Patch(0.2, 1), # a c b                    tota: 3
        Patch(0.3, 2), # a c d b                  tota: 4
        Patch(0.4, 2), # a c e d b                tota: 5
        Patch(0.5, 2), # a c f e d b              tota: 6
        Patch(0.6, 3), # a c f g e d b            tota: 7
        Patch(0.7, 3), # a c f h g e d b          tota: 8
        Patch(0.8, 3), # a c f i h g e d b        tota: 9
        Patch(0.9, 4), # a c f i j h g e d b      tota: 10
    ]

    experiment = Experiment(hyperparameters)

    # # experiment = Experiment()
    experiment.load_tokenizer()
    experiment.load_data()
    experiment.load_model()
    experiment.patch_model()
    experiment.train()
    experiment.model.save_pretrained(OUTPUT_DIR)


    # hyperparameters = HyperParameters()
    # hyperparameters.model_source = 'local'

    # experiment = Experiment(hyperparameters)
    # experiment.load_tokenizer()
    # experiment.load_model()

    # # layer = [experiment.model.bert.encoder.layer[-2], experiment.model.bert.encoder.layer[-1]]
    # # experiment.model.bert.encoder.layer = nn.ModuleList(layer)
    # print(len(experiment.model.bert.encoder.layer))

    # # model.layers = nn.ModuleList([mo]

    # examples = [
    #     "Yesterday I had mac and [MASK] for dinner.",
    #     "I'll be [MASK] back.",
    #     "Paris is the [MASK] of France",
    #     "The city [MASK] is the capital of France",
    #     "Take a [MASK] here, I'll be right back.",
    #     "I hate flying, I have a fear of [MASK].",
    #     "I hate flying, I have a [MASK] of heights.",
    #     "I hate flying, I have a [MASK] of [MASK].",
    #     "Well if it\'s not on top of the [MASK], maybe it fell and it\'s below the [MASK].",
    #     "Well if it\'s not on top of the table, maybe it fell and it\'s [MASK] the table."
    # ]
    # for example in examples:
    #     print(example)
    #     print('  ', experiment.fill_masks(example))