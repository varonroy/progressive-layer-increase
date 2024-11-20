from __future__ import annotations
from typing import Optional, Union, Literal, List, Tuple
from dataclasses import dataclass, field, asdict
from copy import deepcopy

import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertLayer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import get_scheduler
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.integrations import TensorBoardCallback
from datasets import load_dataset, concatenate_datasets


MODEL_NAME = "google-bert/bert-base-uncased"


ModelSource = Optional[Union[Literal['local'], Literal['hf']]]
DataSource = Union[Literal['bookcorpus'], Literal['wikipedia']]


@dataclass
class PatchLayer:
    training_progress: float
    insert_at: int
    clone_from: Optional[int] = None


@dataclass
class PatchNumHeads:
    training_progress: float
    num_heads: int


Patch = Union[PatchLayer, PatchNumHeads]


@dataclass
class HyperParameters:
    model_source: ModelSource = None
    data_sources: List[DataSource] = field(default_factory=lambda: ['bookcorpus', 'wikipedia'])
    # examples: https://huggingface.co/docs/datasets/v1.11.0/splits.html#examples
    data_split: Optional[str] = None
    model_initial_num_layers: Optional[int] = None
    model_initial_num_heads: Optional[int] = None
    model_patches: List[Patch] = field(default_factory=list)
    new_layer_noise: Optional[Tuple[float, float]] = (0.0, 0.01) # (mean, stddev)
    record_grad_norm_every: Optional[int] = 50
    block_size: int = 128
    num_proc: int = 50
    mlm_probability: float = 0.1
    dataset_fraction: Optional[float] = None
    test_size: float = 0.05
    num_train_epochs: int = 1
    lr: float = 2e-5
    lr_end: float = 2e-5 * 2 / 3
    per_device_train_batch_size: int = 32
    warmup_ratio: float = 0.025
    use_tensorboard = True


@dataclass
class CliArgs:
    experiment_name: str
    output_dir: str
    config_path: str
    continue_from: Optional[str] = None


# def load_hyperparameters(config_path: str) -> HyperParameters:
#     with open(config_path, "r") as file:
#         data = json.load(file)
#     return HyperParameters(**data)


# def parse_cli_args():


def load_model_from(source: ModelSource = None, local_source=None):
    if source is None:
        config = AutoConfig.from_pretrained(MODEL_NAME)
        model = AutoModelForMaskedLM.from_config(config)
    elif source == 'local':
        assert local_source is not None
        model = AutoModelForMaskedLM.from_pretrained(local_source)
    elif source == 'hf':
        model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    else:
        raise ValueError(f'model source {source} is not supported')
    assert isinstance(model, BertForMaskedLM)
    return model


def load_data_from(sources: List[DataSource], data_split: Optional[str] = None):
    datasets = []
    split = data_split = "train" if data_split is None else data_split
    for source in sources:
        if source == 'bookcorpus':
            data = load_dataset('bookcorpus', split=split)
        elif source == 'wikipedia':
            # https://huggingface.co/datasets/wikimedia/wikipedia
            data = load_dataset("wikimedia/wikipedia", "20231101.en", split=split)
            remove_columns = list(filter(lambda name: name != 'text', data.column_names))
            data = data.remove_columns(remove_columns)
        else:
            raise ValueError(f'data source {source} is not supported')
        datasets.append(data)

    return concatenate_datasets(datasets)


# # https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling
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


def tokenize_in_chunks(text, tokenizer, block_size):
    """
    Tokenizes text into chunks of size block_size, ensuring no part of the text is discarded.
    """
    chunks = []
    start = 0

    while start < len(text):
        tokenized = tokenizer(
            text[start:],
            truncation=True,
            max_length=block_size,
            return_offsets_mapping=True,
            padding='max_length',
        )

        chunks.append({k: tokenized[k] for k in tokenized if k != 'offset_mapping'})

        offsets = tokenized['offset_mapping']
        if len(offsets) == 0:
            break

        skip = max(o[1] for o in offsets)
        if skip == 0:
            break

        start += skip

    combined = {k: [chunk[k] for chunk in chunks] for k in chunks[0].keys()}

    return combined


def tokenize_in_chunks_batched(texts, tokenizer, block_size):
    """
    Tokenizes text into chunks of size block_size, ensuring no part of the text is discarded.
    """
    chunks = []

    for text in texts:
        start = 0
        while start < len(text):
            tokenized = tokenizer(
                text[start:],
                truncation=True,
                max_length=block_size,
                return_offsets_mapping=True,
                padding='max_length',
            )

            tokenized = {k: v for k, v in tokenized.items() if k != 'offset_mapping'}
            chunks.append(tokenized)

            offsets = tokenized.get('offset_mapping', [])
            if not offsets:
                break

            skip = max(o[1] for o in offsets)
            if skip == 0:
                break

            start += skip

    combined = {k: [chunk[k] for chunk in chunks] for k in chunks[0].keys()}

    return combined


# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_optimizer(optimizer):
    return sum(p.numel() for group in optimizer.param_groups for p in group['params'])


# https://huggingface.co/docs/transformers/v4.46.2/en/main_classes/callback#transformers.TrainerCallback.on_train_begin
class InfoCallback(TensorBoardCallback):
    def __init__(self, h: HyperParameters, cli_args: CliArgs, data):
        super().__init__()
        self.h = h
        self.cli_args = cli_args
        self.data = data

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs['model']

        if self.tb_writer is None:
            self._init_summary_writer(args)

        import json

        self.tb_writer.add_text("cli_args", json.dumps(asdict(self.cli_args), indent=2))

        self.tb_writer.add_text("hyperparameters", json.dumps(asdict(self.h), indent=2))

        self.tb_writer.add_text("model", f"{model}")

        self.tb_writer.add_text("data", f"{self.data}")


# https://discuss.huggingface.co/t/logging-text-using-model-outputs-with-tensorboard/46621/5
class ParameterCountCallback(TensorBoardCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        optimizer = kwargs['optimizer']

        if self.tb_writer is None:
            self._init_summary_writer(args)

        self.tb_writer.add_scalar(
            tag="num_encoder_layers",
            scalar_value=len(model.bert.encoder.layer),
            global_step=state.global_step,
        )

        self.tb_writer.add_scalar(
            tag="num_attention_heads",
            scalar_value=model.config.num_attention_heads,
            global_step=state.global_step,
        )

        self.tb_writer.add_scalar(
            tag="num_encoder_parameters",
            scalar_value=count_parameters(model.bert.encoder.layer),
            global_step=state.global_step,
        )

        self.tb_writer.add_scalar(
            tag="num_optimizer_parameters",
            scalar_value=count_parameters_optimizer(optimizer),
            global_step=state.global_step,
        )


# https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/integrations/integration_utils.py#L629
class GradNormCallback(TensorBoardCallback):
    def __init__(self, record_every: Optional[int]):
        super().__init__()
        self.tb_writer = None
        self.record_every = record_every

    def on_train_begin(self, args, state, control, **kwargs):
        self._init_summary_writer(args)

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self.record_every is None:
            return

        if state.global_step % self.record_every != 0:
            return

        model = kwargs['model']

        for i, l in enumerate(model.bert.encoder.layer):
            grads = []
            for param in l.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))

            if grads:
                grad_vector = torch.cat(grads)
                grad_norm = torch.norm(grad_vector, p=2).item()
            else:
                grad_norm = 0.0

            # Log the gradient norm to TensorBoard
            self.tb_writer.add_scalar(
                f'grad_norms/layer_{i}',
                grad_norm,
                state.global_step,
            )


def modify_linear_features(linear, in_features=None, out_features=None):
    # weight shape is [out_features, in_features]
    # bias shape is [out_features]
    current_out_features = linear.weight.shape[0]
    current_in_features = linear.weight.shape[1]
    weight, bias = linear.weight, linear.bias

    if in_features is not None:
        if weight is not None:
            if in_features < current_in_features:
                weight = linear.weight[:, :in_features]
            else:
                w1 = linear.weight
                w2 = linear.weight[:, :in_features - linear.weight.shape[1]]
                weight = torch.cat([w1, w2], dim=1)

    if out_features is not None:
        if weight is not None:
            if out_features < current_out_features:
                weight = linear.weight[:out_features, :]
            else:
                w1 = linear.weight
                w2 = linear.weight[:out_features - current_out_features, :]
                weight = torch.cat([w1, w2], dim=0)
        if bias is not None:
            if out_features < current_out_features:
                bias = linear.bias[:out_features]
            else:
                b1 = linear.bias[:out_features]
                b2 = linear.bias[:out_features - current_out_features]
                bias = torch.cat([b1, b2], dim=0)
    return weight, bias



def patch_num_heads(model, new_num_h, optimizer=None):
    def remove_param_from_optimizer(param, optimizer):
        if param in optimizer.state:
            del optimizer.state[param]

        for group in optimizer.param_groups:
            params = group['params']
            while True:
                remove = None
                for i, p in enumerate(params):
                    if p is model.bert.encoder.layer[0].attention.self.query.weight:
                        remove = i
                if remove is not None:
                    del params[remove]
                else:
                    break

    def add_param_to_optimizer(param, optimizer):
        optimizer.param_groups[0]['params'].append(param)

    size_h = 64 # bert
    all_head_size = size_h * new_num_h

    model.config.num_attention_heads = new_num_h

    for layer in model.bert.encoder.layer:
        attn = layer.attention.self

        attn.num_attention_heads = new_num_h
        attn.all_head_size = all_head_size

        ### modify attention QKV

        for linear in [attn.query, attn.key, attn.value]:
            if optimizer is not None:
                for param in [linear.weight, linear.bias]:
                    remove_param_from_optimizer(param, optimizer)

            (weight, bias) = modify_linear_features(linear, out_features=all_head_size)
            if weight is not None:
                linear.weight = nn.Parameter(weight)
            if bias is not None:
                linear.bias = nn.Parameter(bias)
            linear.out_features = all_head_size

            if optimizer is not None:
                for param in [linear.weight, linear.bias]:
                    add_param_to_optimizer(param, optimizer)

        ### modify attention output

        attn_output = layer.attention.output
        linear = attn_output.dense

        if optimizer is not None:
            for param in [linear.weight, linear.bias]:
                remove_param_from_optimizer(param, optimizer)

        (weight, bias) = modify_linear_features(linear, in_features=all_head_size)
        if weight is not None:
            linear.weight = nn.Parameter(weight)
        if bias is not None:
            linear.bias = nn.Parameter(bias)
        linear.in_features = all_head_size

        if optimizer is not None:
            for param in [linear.weight, linear.bias]:
                add_param_to_optimizer(param, optimizer)


class PatchingCallback(TrainerCallback):
    def __init__(self, patches: list[Optional[Patch]], new_layer_noise: Optional[Tuple[float, float]], total_training_steps: float):
        self.patches = sorted(patches, key=lambda patch: patch.training_progress)
        self.new_layer_noise = new_layer_noise
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
                patch = self.patches[self.patch_i]

                if patch is not None:
                    model = kwargs['model']
                    optimizer = kwargs['optimizer']
                    config = model.bert.encoder.config

                    if isinstance(patch, PatchLayer):
                        layer = list(model.bert.encoder.layer)

                        new_block = (
                            deepcopy(layer[patch.clone_from]) if patch.clone_from is not None else BertLayer(config)
                        ).to(model.device)

                        if self.new_layer_noise is not None:
                            (mean, stddev) = self.new_layer_noise
                            for p in new_block.parameters():
                                noise = torch.rand_like(p) * stddev + mean
                                p.add_(noise)

                        layer_list = list(layer)
                        layer_list.insert(patch.insert_at, new_block)
                        layer = nn.ModuleList(layer_list)

                        optimizer.add_param_group({"params": new_block.parameters()})

                        model.bert.encoder.layer = nn.ModuleList(layer)
                        config.num_hidden_layers = len(layer)
                        assert model.config is config, "config pointers don't match"
                    elif isinstance(patch, PatchNumHeads):
                        num_heads = patch.num_heads
                        patch_num_heads(model, num_heads, optimizer)
                    else:
                        raise ValueError(f'invalid patch type {type(patch)}')

                self.patch_i += 1


class Experiment:
    def __init__(self, cli_args: CliArgs, hyperparameters: Optional[HyperParameters] = None):
        self.cli_args = cli_args
        if hyperparameters is None:
            self.hyperparameters = HyperParameters()
        else:
            self.hyperparameters = hyperparameters

    def load_tokenizer(self):
        # https://huggingface.co/google-bert/bert-base-uncased?library=transformers
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        assert isinstance(self.tokenizer, BertTokenizerFast)

    def load_model(self):
        cli = self.cli_args
        h = self.hyperparameters
        self.model = load_model_from(
            h.model_source,
            cli.continue_from if cli.continue_from is not None else cli.output_dir,
        )

    def load_data(self):
        h = self.hyperparameters
        assert self.tokenizer is not None

        data = load_data_from(h.data_sources, h.data_split)

        data = data.map(
            # lambda item: self.tokenizer(item['text']),
            # lambda item: tokenize_in_chunks(item['text'], self.tokenizer, h.block_size),
            lambda item: tokenize_in_chunks_batched(item['text'], self.tokenizer, h.block_size),
            batched=True,
            num_proc=h.num_proc,
            remove_columns=data.column_names,
        )

        if h.dataset_fraction is not None:
            new_size = int(data.num_rows * h.dataset_fraction)
            data = data.select(range(new_size))

        data = data.train_test_split(test_size=h.test_size)

        # data = data.map(
        #     lambda item: group_texts(h.block_size, item),
        #     batched=True,
        #     num_proc=h.num_proc,
        # )

        data = data.map(
            lambda item: item,
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

        if h.model_initial_num_layers is not None:
            layer = [
                BertLayer(self.model.bert.encoder.config)
                for _ in range(h.model_initial_num_layers)
            ]
            self.model.bert.encoder.layer = nn.ModuleList(layer)

        # When initializing bert, the hidden size should be a multiple of the number of heads.
        # Therefore, all changes to the heads must be done afer creating the layers.
        if h.model_initial_num_heads is not None:
            patch_num_heads(self.model, h.model_initial_num_heads)

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
            output_dir=self.cli_args.output_dir,
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
            info_callback = InfoCallback(h, self.cli_args, self.data['data'])
            trainer.add_callback(info_callback)

            patch_callback = PatchingCallback(
                patches=h.model_patches,
                new_layer_noise=h.new_layer_noise,
                total_training_steps=self.total_training_steps(),
            )
            trainer.add_callback(patch_callback)

            parameter_callback = ParameterCountCallback()
            trainer.add_callback(parameter_callback)

            grad_norm_callback = GradNormCallback(record_every=h.record_grad_norm_every)
            trainer.add_callback(grad_norm_callback)

        return trainer

    def train(self):
        trainer = self.build_trainer()
        trainer.train()
        self.trainer = trainer

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


def sanity_check(data_slice=50000):
    cli_args = CliArgs(
        experiment_name="sanity-check",
        output_dir="sanity-check",
        config_path=None,
        continue_from=None
    )

    hyperparameters = HyperParameters(
        # data
        data_sources=['bookcorpus', 'wikipedia'],
        data_split=f"train[:{data_slice}]" if data_slice is not None else "train",
        # leraning rate
        lr=1.0e-4,
        lr_end=0.0,
        # model
        model_initial_num_heads=4,
        model_initial_num_layers=1,
        model_patches=[
            # layers
            *[PatchLayer((i + 1) / 12, 0, 0) for i in range(11)],
            # attention heads
            PatchNumHeads(0.2, 8),
            PatchNumHeads(0.4, 12),
        ],
        # train
        num_train_epochs=2
    )

    experiment = Experiment(cli_args, hyperparameters)

    experiment.load_tokenizer()
    experiment.load_data()
    experiment.load_model()
    experiment.patch_model()
    experiment.train()
    experiment.model.save_pretrained(cli_args.output_dir)


def main_increase():
    cli_args = CliArgs(
        experiment_name="main-experiment",
        output_dir="bert-pretraining",
        config_path=None,
        continue_from=None
    )

    hyperparameters = HyperParameters(
        # data
        data_sources=['bookcorpus', 'wikipedia'],
        # model
        model_initial_num_heads=4,
        model_initial_num_layers=1,
        model_patches=[
            # layers
            *[PatchLayer((i + 1) / 12, 0, 0) for i in range(11)],
            # attention heads
            PatchNumHeads(0.2, 8),
            PatchNumHeads(0.4, 12),
        ],
        # train
        num_train_epochs=2
    )

    experiment = Experiment(cli_args, hyperparameters)

    experiment.load_tokenizer()
    experiment.load_data()
    experiment.load_model()
    experiment.patch_model()
    experiment.train()
    experiment.model.save_pretrained(cli_args.output_dir)


def main_regular():
    cli_args = CliArgs(
        experiment_name="main-experiment-regular",
        output_dir="bert-pretraining-regular",
        config_path=None,
        continue_from=None
    )

    hyperparameters = HyperParameters(
        # data
        data_sources=['bookcorpus', 'wikipedia'],
        # # model
        # model_initial_num_heads=4,
        # model_initial_num_layers=1,
        # model_patches=[
        #     # layers
        #     *[PatchLayer((i + 1) / 12, 0, 0) for i in range(11)],
        #     # attention heads
        #     PatchNumHeads(0.2, 8),
        #     PatchNumHeads(0.4, 12),
        # ],
        # train
        num_train_epochs=2
    )

    experiment = Experiment(cli_args, hyperparameters)

    experiment.load_tokenizer()
    experiment.load_data()
    experiment.load_model()
    experiment.patch_model()
    experiment.train()
    experiment.model.save_pretrained(cli_args.output_dir)


def main_increase_v2():
    cli_args = CliArgs(
        experiment_name="main-experiment-v2",
        output_dir="bert-pretraining-v2",
        config_path=None,
        continue_from=None
    )

    hyperparameters = HyperParameters(
        # data
        data_sources=['bookcorpus', 'wikipedia'],
        # leraning rate
        lr=1.0e-4,
        lr_end=0.0,
        # model
        model_initial_num_layers=1,
        model_patches=[
            # layers
            *[PatchLayer((i + 1) / 12, 0, 0) for i in range(11)],
        ],
        # train
        num_train_epochs=1
    )

    experiment = Experiment(cli_args, hyperparameters)

    experiment.load_tokenizer()
    experiment.load_data()
    experiment.load_model()
    experiment.patch_model()
    experiment.train()
    experiment.model.save_pretrained(cli_args.output_dir)


def run_different_placements():
    for experiment_name, output_dir, model_patches in [
        (
            "testing-placements-start",
            "testing-placements/start",
            [PatchLayer((i + 1) / 5, 0, 0) for i in range(4)],
        ),
        (
            "testing-placements-end",
            "testing-placements/end",
            [PatchLayer((i + 1) / 5, i, i) for i in range(4)],
        ),
        (
            "testing-placements-middle",
            "testing-placements/middle",
            [PatchLayer((i + 1) / 5, i // 2, i // 2) for i in range(4)],
        ),
        (
            "testing-placements-start-bulk",
            "testing-placements/start-bulk",
            [
                PatchLayer(1 / 6, 0, 0),
                PatchLayer(1 / 6, 0, 0),
                PatchLayer(3 / 6, 0, 0),
                PatchLayer(3 / 6, 0, 0),
            ],
        ),
        (
            "testing-placements-end-bulk",
            "testing-placements/end-bulk",
            [
                PatchLayer(1 / 6, 1, 1),
                PatchLayer(1 / 6, 2, 2),
                PatchLayer(3 / 6, 3, 3),
                PatchLayer(3 / 6, 4, 4),
            ],
        ),
        (
            "testing-placements-middle-bulk",
            "testing-placements/middle-bulk",
            [
                PatchLayer(1 / 6, 0, 0),
                PatchLayer(1 / 6, 0, 0),
                PatchLayer(3 / 6, 1, 1),
                PatchLayer(3 / 6, 1, 1),
            ],
        ),
    ]:
        cli_args = CliArgs(
            experiment_name=experiment_name,
            output_dir=output_dir,
            config_path=None,
            continue_from=None
        )

        hyperparameters = HyperParameters(
            # data
            data_sources=['wikipedia'],
            data_split="train[:50000]",
            # leraning rate
            lr=1.0e-4,
            lr_end=0.0,
            # model
            model_initial_num_layers=1,
            model_patches=model_patches,
            # train
            num_train_epochs=1
        )

        experiment = Experiment(cli_args, hyperparameters)

        experiment.load_tokenizer()
        experiment.load_data()
        experiment.load_model()
        experiment.patch_model()
        experiment.train()
        experiment.model.save_pretrained(cli_args.output_dir)

if __name__ == '__main__':
    sanity_check(data_slice=None)


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