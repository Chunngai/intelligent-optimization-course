import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ant import Ant

np.set_printoptions(threshold=sys.maxsize)


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def exact_agreement(trgs: np.ndarray, outs: np.ndarray) -> float:
    assert trgs.shape == outs.shape

    n_match = np.sum(
        trgs == outs
    )
    n_all = trgs.shape[0]

    res = n_match / n_all
    return res


def adjacent_agreement(trgs: np.ndarray, outs: np.ndarray) -> float:
    assert trgs.shape == outs.shape

    differences = trgs - outs
    n_match = np.sum(
        np.abs(differences) <= 1
    )
    n_all = trgs.shape[0]

    res = n_match / n_all
    return res


class ACOFeatureSelector:

    def __init__(
            self, fp_data: str, n_features_to_select: int, model_class,
            n_ants: int, n_iters: int,
            alpha: int = 1, beta: int = 1, q: int = 1, mu: float = 1.0, rho: float = 0.1,
    ):
        # Read data.
        df = pd.read_csv(fp_data)
        df = df.loc[:, ~(df == df.iloc[0]).all()]  # Remove columns with identical values (0, 4, 43, 44).
        df = df.to_numpy()
        classes = df[:, -1].astype(int)
        df = np.delete(df, -1, 1)

        # Train/test split.
        (
            self.training_xs, self.testing_xs,
            self.training_ys, self.testing_ys
        ) = train_test_split(
            df,
            classes,
            random_state=42,
        )

        self.n_all_features = len(self.training_xs[0])
        self.all_features = np.arange(self.n_all_features)

        self.n_features_to_select = n_features_to_select
        if self.n_features_to_select > self.n_all_features:
            self.n_features_to_select = self.n_all_features

        # Scale features.
        scaler = StandardScaler().fit(self.training_xs)
        self.training_xs = scaler.transform(self.training_xs)
        self.testing_xs = scaler.transform(self.testing_xs)

        self.model_class = model_class

        self.n_ants = n_ants
        self.n_iters = n_iters

        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.mu = mu
        self.rho = rho

        self.ants = None  # TODO
        self.ant_accuracies = np.zeros(self.n_ants)  # TODO

        self.feature_pheromone = np.full(
            shape=self.n_all_features,
            fill_value=self.mu,
        )

        self.lut = None  # TODO

    def define_lut(self):
        """Defines the Look-Up Table (LUT) for the algorithm.
        """

        kbest = SelectKBest(
            score_func=f_classif,
            k='all'
        )
        kbest.fit(self.training_xs, self.training_ys)

        # Normalization.
        self.lut = kbest.scores_
        sum_ = np.sum(self.lut)
        for i in range(len(kbest.scores_)):
            self.lut[i] = self.lut[i] / sum_

    def redefine_lut(self, feature):
        """Re-defines the Look-Up Table (LUT) for the algorithm.
        """

        weight_prob = self.lut[feature]
        self.lut[feature] = 0
        mult = 1 / (1 - weight_prob)
        self.lut = self.lut * mult

    def reset_initial_values(self, ant_index):
        """Initialize the ant array and assign each one a random initial feature.
        """

        random_feature = np.random.choice(
            self.all_features,
            size=1,
            p=self.lut
        )[0]
        self.ants[ant_index].feature_path.append(random_feature)

        ant_features = self.ants[ant_index].feature_path
        ant_xs = np.array(self.training_xs[:, ant_features])

        model = self.model_class()
        model.fit(
            X=ant_xs,
            y=self.training_ys,
        )
        scores = cross_val_score(
            estimator=model,
            X=ant_xs,
            y=self.training_ys,
            cv=5,
        )
        ant_acc = scores.mean()

        np.put(
            self.ant_accuracies,
            ant_index,
            ant_acc,
        )

    def ant_build_subset(self, ant_index):
        """Global and local search for the ACO algorithm. It completes the subset of features of the ant searching.

        :param ant_index: Ant that is going to do the local search.
        :type ant_index: Integer
        """

        self.define_lut()
        self.reset_initial_values(ant_index=ant_index)

        # Initialize unvisited features and it removes the first of the ant actual subset
        visited_features = np.where(np.in1d(
            self.all_features,
            self.ants[ant_index].feature_path,
        ))[0]
        unvisited_features = np.delete(
            self.all_features,
            visited_features,
        )

        self.redefine_lut(feature=visited_features[0])

        for _ in range(self.n_features_to_select):
            # Compute eta, tau and the numerator for each unvisited feature
            probs = np.zeros(np.size(unvisited_features))
            for i, unvisited_feature in enumerate(unvisited_features):
                eta = self.lut[unvisited_feature]
                tau = self.feature_pheromone[unvisited_feature]

                np.put(
                    probs,
                    i,
                    (tau ** self.alpha) * (eta ** self.beta),
                )

            sum_ = np.sum(probs)
            for i in range(len(unvisited_features)):
                probs[i] /= sum_
            next_feature = np.random.choice(
                unvisited_features,
                size=1,
                p=probs,
            )[0]

            # Choose the feature with best probability and add to the ant subset
            self.ants[ant_index].feature_path.append(next_feature)
            # Remove the chosen feature of the unvisited features
            unvisited_features = np.delete(
                unvisited_features,
                np.where(unvisited_features == next_feature),
            )

            self.redefine_lut(feature=next_feature)

        ant_features = np.array(self.ants[ant_index].feature_path)
        ant_xs = np.array(self.training_xs[:, ant_features])

        model = self.model_class()
        model.fit(
            X=ant_xs,
            y=self.training_ys,
        )
        scores = cross_val_score(
            estimator=model,
            X=ant_xs,
            y=self.training_ys,
            cv=5,
        )
        new_ant_acc = scores.mean()

        np.put(
            self.ant_accuracies,
            ant_index,
            new_ant_acc,
        )

    def update_pheromones(self):
        """Update the pheromones trail depending on which variant of the algorithm it is selected.
        """

        best_ant_index = np.argmax(self.ant_accuracies)
        best_acc = np.max(self.ant_accuracies)
        for selected_feature in self.ants[best_ant_index].feature_path:
            sum_delta = self.q / ((1 - best_acc) * 100)

            updated_pheromone = (1 - self.rho) * self.feature_pheromone[selected_feature] + sum_delta
            if updated_pheromone < 0.4:
                updated_pheromone = 0.4

            np.put(
                self.feature_pheromone,
                selected_feature,
                updated_pheromone,
            )

    def select_features(self):
        for _ in tqdm(range(self.n_iters)):
            self.ants = [Ant() for _ in range(self.n_ants)]
            for ant_index in range(self.n_ants):
                self.ant_build_subset(ant_index)
            self.update_pheromones()

        return self.evaluate()

    def evaluate(self):
        """Function for printing the entire summary of the algorithm, including the test results.
        """

        selected_features = self.ants[np.argmax(self.ant_accuracies)].feature_path
        # print("The final subset of features is: ", selected_features)
        # print("Number of features: ", len(selected_features))

        model_before = self.model_class()
        model_before.fit(self.training_xs, self.training_ys)
        outputs_before = model_before.predict(self.testing_xs)

        exact_agreement_before = exact_agreement(
            trgs=self.testing_ys,
            outs=outputs_before,
        )
        adjacent_agreement_before = adjacent_agreement(
            trgs=self.testing_ys,
            outs=outputs_before,
        )
        qwk_before = cohen_kappa_score(
            y1=self.testing_ys,
            y2=outputs_before,
            weights="quadratic",
        )
        correlation_before = pearsonr(
            x=self.testing_ys,
            y=outputs_before,
        )[0]

        data_training_subset = self.training_xs[:, self.ants[np.argmax(self.ant_accuracies)].feature_path]
        data_testing_subset = self.testing_xs[:, self.ants[np.argmax(self.ant_accuracies)].feature_path]

        model_after = self.model_class()
        model_after.fit(data_training_subset, self.training_ys)
        outputs_after = model_after.predict(data_testing_subset)

        exact_agreement_after = exact_agreement(
            trgs=self.testing_ys,
            outs=outputs_after,
        )
        adjacent_agreement_after = adjacent_agreement(
            trgs=self.testing_ys,
            outs=outputs_after,
        )
        qwk_after = cohen_kappa_score(
            y1=self.testing_ys,
            y2=outputs_after,
            weights="quadratic",
        )
        correlation_after = pearsonr(
            x=self.testing_ys,
            y=outputs_after,
        )[0]

        return {
            "qwk_before": qwk_before,
            "exact_agreement_before": exact_agreement_before,
            "adjacent_agreement_before": adjacent_agreement_before,
            "correlation_before": correlation_before,

            "qwk_after": qwk_after,
            "exact_agreement_after": exact_agreement_after,
            "adjacent_agreement_after": adjacent_agreement_after,
            "correlation_after": correlation_after,

            "selected_features": selected_features,
        }


if __name__ == '__main__':
    models = (
        GaussianNB,
        # KNeighborsClassifier,  # 30 ants
        # DecisionTreeClassifier,  # 30 ants
        # LogisticRegression,
        # SVC,
        # SGDClassifier,
    )
    essay_sets = range(1, 8 + 1)

    for essay_set in essay_sets:
        print(f"essay set: {essay_set}")
        for model in models:

            rst_dicts = {}
            for i in range(3):
                feature_selector = ACOFeatureSelector(
                    fp_data=f"data/data.{essay_set}.csv",
                    n_features_to_select=30,
                    model_class=model,
                    n_ants=20,
                    n_iters=30,
                    alpha=1,
                    beta=1,
                    q=3,
                    mu=1.0,
                    rho=0.6,
                )
                rst_dict = feature_selector.select_features()
                print(f"\t[mdl] {model.__name__}")
                print(
                    f"\t[bfr] "
                    f"qwk={rst_dict['qwk_before']:.2f}, "
                    f"ea={rst_dict['exact_agreement_before']:.2f}, "
                    f"aa={rst_dict['adjacent_agreement_before']:.2f}, "
                    f"corr={rst_dict['correlation_before']:.2f}"
                )
                print(
                    f"\t[aft] "
                    f"qwk={rst_dict['qwk_after']:.2f}, "
                    f"ea={rst_dict['exact_agreement_after']:.2f}, "
                    f"aa={rst_dict['adjacent_agreement_after']:.2f}, "
                    f"corr={rst_dict['correlation_after']:.2f}"
                )
                print()

                rst_dicts[i] = rst_dict

            with open(f"{essay_set}_{model.__name__}.txt", "w", encoding="utf-8") as f:
                f.write(str(rst_dicts))
