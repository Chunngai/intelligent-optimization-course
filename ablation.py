import csv
import traceback

from sklearn.naive_bayes import GaussianNB

from main import ACOFeatureSelector

for i in range(10):
    with open(f"ablation.{i}.csv", "w", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(
            f,
            fieldnames=("n_ants", "n_iters", "alpha", "beta", "q", "rho", "qwk_before", "qwk_after", "qwk_delta")
        )
        csv_writer.writeheader()

        default_hyperparams = {
            "n_ants": 10,
            "n_iters": 10,
            "alpha": 1,
            "beta": 1,
            "q": 1,
            "rho": 0.1,
        }

        hyperparam_choices = {
            "n_ants": range(10, 100 + 1, 10),
            "n_iters": range(10, 100 + 1, 10),
            "alpha": range(1, 5 + 1),
            "beta": range(1, 5 + 1),
            "q": range(1, 10 + 1),
            "rho": list(map(lambda val: val / 10, range(1, 10, 1))),
        }

        for hyperparam, choices in hyperparam_choices.items():
            for choice in choices:
                d = dict(default_hyperparams)
                d[hyperparam] = choice
                print(d)

                feature_selector = ACOFeatureSelector(
                    fp_data=f"data/data.1.csv",
                    n_features_to_select=30,
                    model_class=GaussianNB,
                    mu=1.0,
                    **d,
                )

                try:
                    qwks = feature_selector.select_features()
                    print(f"\t[bfr] {qwks['qwk_before']}")
                    print(f"\t[aft] {qwks['qwk_after']}")
                    print(f"\t[dlt] {qwks['qwk_after'] - qwks['qwk_before']}")
                    print()
                except:
                    print(traceback.format_exc())
                    qwks = {
                        "qwk_before": float("nan"),
                        "qwk_after": float("nan"),
                    }

                d["qwk_before"] = qwks["qwk_before"]
                d["qwk_after"] = qwks["qwk_after"]
                d["qwk_delta"] = qwks["qwk_after"] - qwks["qwk_before"]

                csv_writer.writerow(d)
