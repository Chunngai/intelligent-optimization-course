import csv


def read_csv_rows(fp: str) -> list:
    with open(fp, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f)
        return list(csv_reader)


asap_train = read_csv_rows(fp="data/fold_0/ext_asap_train.csv")
asap_dev = read_csv_rows(fp="data/fold_0/ext_asap_dev.csv")
asap_test = read_csv_rows(fp="data/fold_0/ext_asap_test.csv")

asap_fields = asap_train[0]
asap_lines = asap_train[1:] + asap_dev[1:] + asap_test[1:]
print(len(asap_lines))

features_train = read_csv_rows(fp="data/fold_0/features_train.csv")
features_dev = read_csv_rows(fp="data/fold_0/features_dev.csv")
features_test = read_csv_rows(fp="data/fold_0/features_test.csv")

features_fields = features_train[0]
features_lines = features_train[1:] + features_dev[1:] + features_test[1:]
print(len(features_lines))

for essay_set in range(1, 8 + 1):
    with open(f"data/coh-metrix.{essay_set}.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for asap_line, feature_line in zip(asap_lines, features_lines):
            if asap_line[1] == str(essay_set):
                writer.writerow(feature_line[2:] + [asap_line[3]])
