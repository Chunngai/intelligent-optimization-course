import csv
import traceback

from sklearn.naive_bayes import GaussianNB

from main import ACOFeatureSelector

for i in range(5):
    with open(f"ablation.{i}.csv", "w", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(
            f,
            fieldnames=("n_ants", "n_iters", "alpha", "beta", "q", "rho", "qwk_before", "qwk_after", "qwk_delta")
        )
        csv_writer.writeheader()

        for n_ants in [10, 20, 30]:
            for n_iters in [10, 30, 50]:
                for alpha in [1, 2, 3]:
                    for beta in [1, 2, 3, 4, 5]:
                        for q in [1, 2, 3]:
                            for rho in [0.1, 0.3, 0.5, 0.7, 0.9]:

                                line = {
                                    "n_ants": n_ants,
                                    "n_iters": n_iters,
                                    "alpha": alpha,
                                    "beta": beta,
                                    "q": q,
                                    "rho": rho,
                                }
                                print(line)

                                feature_selector = ACOFeatureSelector(
                                    fp_data=f"data/data.1.csv",
                                    n_selected_features=30,
                                    model_class=GaussianNB,
                                    initial_pheromone=1.0,
                                    **line,
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

                                line["qwk_before"] = qwks["qwk_before"]
                                line["qwk_after"] = qwks["qwk_after"]
                                line["qwk_delta"] = qwks["qwk_after"] - qwks["qwk_before"]

                                csv_writer.writerow(line)
