import json
import matplotlib.pyplot as plt

# files = [
#     ["./data/layers-2.json", "2-layer"],
#     ["./data/layers-4.json", "4-layer"],
#     ["./data/layers-8.json", "8-layer"],
# ]

files = [
    ["./data/layers-12-higher-lr.json", "12-layer"],
    ["./data/layers-1-10-PLI-higher-lr.json", "1-10-layers PLI"],
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


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
xoffset = 10


def plot_data_entry(entry, x_ty):
    plt.plot(entry[x_ty][xoffset:], entry["loss"][xoffset:], label=entry["name"])


def plot_with_x_ty(x_ty, ax, title):
    plt.sca(ax)
    for entry in data.values():
        plot_data_entry(entry, x_ty)
    plt.xlabel(title)
    plt.ylabel("Loss")
    plt.title(f"Loss vs {title}")
    plt.legend()
    plt.grid(True)


plot_with_x_ty("iteration", ax1, "Iteration")
plot_with_x_ty("time", ax2, "Time")


plt.show()
