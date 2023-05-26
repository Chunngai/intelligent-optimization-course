import csv

import numpy as np

all_lines = None
fields = None
for i in range(10):
    with open(f"ablation.{i}.csv", "r", encoding="utf-8") as f:
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