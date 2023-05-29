import csv

import matplotlib.pyplot as plt
import numpy as np

all_lines = None
fields = None
for i in range(10):
    with open(f"ablation/ablation.{i}.csv", "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f)
        lines = list(csv_reader)

        fields = lines[0]
        lines = lines[1:]

        lines = np.array(lines, dtype=float)

        if all_lines is None:
            all_lines = lines
        else:
            all_lines += lines

all_lines /= 10

for line in all_lines:
    for field, val in zip(fields, line):
        print(f"{field}: {val:.3f}", end=" ")
    print()


def visualize(var_idx: int, start_line_idx: int, end_line_idx: int, var_name: str, xlabel: str, ylabel: str,
              xtick_step: float):
    lines = all_lines[start_line_idx:end_line_idx + 1]

    xs = list(map(
        lambda line: line[var_idx],
        lines
    ))
    ys = list(map(
        lambda line: line[-1],
        lines
    ))

    plt.plot(xs, ys)
    plt.xticks(np.arange(min(xs), max(xs) + (xs[1] - xs[0]), xtick_step))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0.05, 0.11)
    plt.savefig(f"ablation/ablation_{var_name}.png")


visualize(
    var_idx=0,
    start_line_idx=0,
    end_line_idx=9,
    var_name="n_ants",
    xlabel="Ant Number",
    ylabel="QWK Improvement",
    xtick_step=10,
)
plt.clf()

visualize(
    var_idx=1,
    start_line_idx=10,
    end_line_idx=19,
    var_name="n_iters",
    xlabel="Colony Number",
    ylabel="QWK Improvement",
    xtick_step=10,
)
plt.clf()

visualize(
    var_idx=2,
    start_line_idx=20,
    end_line_idx=24,
    var_name="alpha",
    xlabel="Alpha",
    ylabel="QWK Improvement",
    xtick_step=1,
)
plt.clf()

visualize(
    var_idx=3,
    start_line_idx=25,
    end_line_idx=29,
    var_name="beta",
    xlabel="Beta",
    ylabel="QWK Improvement",
    xtick_step=1,
)
plt.clf()

visualize(
    var_idx=4,
    start_line_idx=30,
    end_line_idx=39,
    var_name="q",
    xlabel="Q",
    ylabel="QWK Improvement",
    xtick_step=1,
)
plt.clf()

visualize(
    var_idx=5,
    start_line_idx=40,
    end_line_idx=49,
    var_name="rho",
    xlabel="Evaporation Rate",
    ylabel="QWK Improvement",
    xtick_step=0.2,
)
plt.clf()
