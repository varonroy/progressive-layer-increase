import json
import matplotlib.pyplot as plt

target = 'main-experiment-loss'

if target == 'layers':
    xoffset = 10
    plot_target = 'both'
    num_steps = None
    files = [
        ["./data-intro/layers-2.json", "2-layer"],
        ["./data-intro/layers-4.json", "4-layer"],
        ["./data-intro/layers-8.json", "8-layer"],
    ]

if target == 'lr':
    xoffset = 10
    plot_target = 'both'
    num_steps = None
    files = [
        ["./data-intro/layers-12-higher-lr.json", "12-layer"],
        ["./data-intro/layers-1-10-PLI-higher-lr.json", "1-10-layers PLI"],
    ]

if target == 'placement':
    xoffset = 0
    plot_target = 'iteration'
    num_steps = None
    files = [
        ["./data-layer-placement/testing-placements_end_runs_Nov16_10-56-15_dev.json", "end"],
        ["./data-layer-placement/testing-placements_start-bulk_runs_Nov16_11-30-27_dev.json", "start-bulk"],
        ["./data-layer-placement/testing-placements_start_runs_Nov16_10-39-04_dev.json", "start"],
        ["./data-layer-placement/testing-placements_middle_runs_Nov16_11-13-22_dev.json", "middle"],
    ]

if target == 'pretraining':
    xoffset = 50
    plot_target = 'iteration'
    num_steps = 100
    files = [
        ["./data-pretraining/bert-pretraining-v2_runs_Nov09_07-49-52_dev.json", "PLI - higher-lr"],
        ["./data-pretraining/bert-pretraining-v2_runs_Nov09_13-06-29_dev.json", "PLI - lower-lr"],
        # ["./data-pretraining/bert-pretraining-v2_runs_Oct29_19-53-40_dev.json", "full model"],
    ]

if target == 'main-experiment-loss':
    xoffset = 50
    plot_target = 'time'
    num_steps = None
    files = [
        ["./data-main-experiment/bert-pretraining-regular_runs_Nov03_19-46-51_dev-loss.json", "Full model"],
        ["./data-main-experiment/bert-pretraining-v2_runs_Nov09_13-06-29_dev-loss.json", "PLI"],
    ]

if target == 'main-experiment-num-layers':
    xoffset = 50
    plot_target = 'iteration'
    num_steps = None
    files = [
        ["./data-main-experiment/bert-pretraining-v2_runs_Nov03_19-46-51_dev-num-layers.json", "Full model"],
        ["./data-main-experiment/bert-pretraining-v2_runs_Nov09_13-06-29_dev-num-layers.json", "PLI"],
    ]


# To store the data for plotting
data = {}


# Loop through each file and extract the necessary data
for entry in files:
    file_name, name = entry[0], entry[1]

    with open(file_name) as f:
        json_data = json.load(f)

    time0 = json_data[0][0]
    time = [entry[0] - time0 for entry in json_data]

    iteration = [entry[1] for entry in json_data]

    loss = [entry[2] for entry in json_data]

    data[name] = {
        "name": name,
        "time": time,
        "iteration": iteration,
        "loss": loss,
    }


def plot_data_entry(entry, x_ty):
    x = entry[x_ty][xoffset:]
    y = entry["loss"][xoffset:]
    if num_steps != None:
        steps = min(num_steps, len(x), len(y))
        x = x[:steps]
        y = y[:steps]
    plt.plot(x, y, label=entry["name"])


def plot_with_x_ty(x_ty, ax, title):
    plt.sca(ax)
    for entry in data.values():
        plot_data_entry(entry, x_ty)
    plt.xlabel(title)
    plt.ylabel("Loss")
    plt.title(f"Loss vs {title}")
    plt.legend()
    plt.grid(True)

if plot_target == 'both':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    axs = [ax1, ax2]
else:
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    axs = [ax]

ax_i = 0

if plot_target == "iteration" or plot_target == 'both':
    plot_with_x_ty("iteration", axs[ax_i], "Iteration")
    ax_i += 1

if plot_target == "time" or plot_target == 'both':
    plot_with_x_ty("time", axs[ax_i], "Time")
    ax_i += 1


plt.show()
