__author__ = 'Alberto Ortega'
__copyright__ = 'Pathfinder (c) 2021 EFFICOMP'
__credits__ = 'Spanish Ministerio de Ciencia, Innovacion y Universidades under grant number PGC2018-098813-B-C31. European Regional Development Fund (ERDF).'
__license__ = ' GPL-3.0'
__version__ = "2.0"
__maintainer__ = 'Alberto Ortega'
__email__ = 'aoruiz@ugr.es'

import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from ant import Ant

np.set_printoptions(threshold=sys.maxsize)


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def exact_agreement(trgs, outs):
    assert trgs.shape == outs.shape

    n_match = np.sum(trgs == outs)
    n_all = trgs.shape[0]

    res = n_match / n_all
    return res


def adjacent_agreement(trgs, outs):
    assert trgs.shape == outs.shape

    differences = trgs - outs
    n_match = np.sum((np.abs(differences) <= 1))
    n_all = trgs.shape[0]

    res = n_match / n_all
    return res


class ACOFeatureSelector:
    """Class for Ant System Optimization algorithm designed for Feature Selection.

    :param dtype: Format of the dataset.
    :param data_training_name: Path to the training data file (mat) or path to the dataset file (csv).
    :param class_training: Path to the training classes file (mat).
    :param data_testing: Path to the testing data file (mat).
    :param class_testing: Path to the testing classes file (mat).
    :param n_ants: Number of ants of the colonies.
    :param n_iters: Number of colonies of the algorithm.
    :param n_selected_features: Number of features to be selected.
    :param alpha: Parameter which determines the weight of tau.
    :param beta: Parameter which determines the weight of eta.
    :param q: Parameter for the pheromones update function.
    :param initial_pheromone: Initial value for the pheromones.
    :param rho: Rate of the pheromones evaporation.
    :type dtype: MAT or CSV
    :type data_training_name: Numpy array
    :type training_ys: Numpy array
    :type data_testing: Numpy array
    :type testing_ys: Numpy array
    :type n_ants: Integer
    :type n_iters: Integer
    :type n_selected_features: Integer
    :type alpha: Float
    :type beta: Float
    :type q: Float
    :type initial_pheromone: Float
    :type rho: Float

    """

    def __init__(
            self, fp_data: str, n_selected_features: int, model_class,
            n_ants: int, n_iters: int,
            alpha: int = 1, beta: int = 1, q: int = 1, initial_pheromone: float = 1.0, rho: float = 0.1,
    ):
        # Read data.
        df = pd.read_csv(fp_data)
        df = df.loc[:, ~(df == df.iloc[0]).all()]  # Remove columns with identical values.
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
            random_state=42
        )
        self.n_all_features = len(self.training_xs[0])

        self.n_selected_features = n_selected_features
        if self.n_selected_features > self.n_all_features:
            self.n_selected_features = self.n_all_features

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
        self.initial_pheromone = initial_pheromone
        self.rho = rho

        self.ants = None
        self.ant_accuracy_list = np.zeros(self.n_ants)
        self.unvisited_feature_indexes = None

        self.feature_pheromone = np.full(self.n_all_features, self.initial_pheromone)

        self.lut = None

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

    def reset_initial_values(self):
        """Initialize the ant array and assign each one a random initial feature.
        """

        self.ants = [Ant() for _ in range(self.n_ants)]
        all_feature_indexes = np.arange(self.n_all_features)
        for i in range(self.n_ants):
            random_feature_index = np.random.choice(all_feature_indexes, 1, p=self.lut)[0]
            self.ants[i].feature_path.append(random_feature_index)

            ant_feature_indexes = self.ants[i].feature_path
            ant_xs = np.array(self.training_xs[:, ant_feature_indexes])

            model = self.model_class()
            model.fit(ant_xs, self.training_ys)
            scores = cross_val_score(model, ant_xs, self.training_ys, cv=5)
            ant_acc = scores.mean()

            np.put(self.ant_accuracy_list, i, ant_acc)

    def ant_build_subset(self, index_ant):
        """Global and local search for the ACO algorithm. It completes the subset of features of the ant searching.

        :param index_ant: Ant that is going to do the local search.
        :type index_ant: Integer
        """

        # Initialize unvisited features and it removes the first of the ant actual subset
        all_feature_indexes = np.arange(self.n_all_features)
        visited_feature_indexes = np.where(np.in1d(
            all_feature_indexes,
            self.ants[index_ant].feature_path
        ))[0]
        self.unvisited_feature_indexes = np.delete(all_feature_indexes, visited_feature_indexes)

        self.define_lut()

        for n in range(self.n_selected_features):
            # Compute eta, tau and the numerator for each unvisited feature
            probs = np.zeros(np.size(self.unvisited_feature_indexes))
            for unvisited_feature_index in range(len(self.unvisited_feature_indexes)):
                eta = self.lut[self.unvisited_feature_indexes[unvisited_feature_index]]
                tau = self.feature_pheromone[unvisited_feature_index]

                np.put(probs, unvisited_feature_index, (tau ** self.alpha) * (eta ** self.beta))

            sum_ = np.sum(probs)
            for unvisited_feature_index in range(len(self.unvisited_feature_indexes)):
                probs[unvisited_feature_index] = probs[unvisited_feature_index] / sum_
            next_feature = np.random.choice(self.unvisited_feature_indexes, 1, p=probs)[0]

            # Choose the feature with best probability and add to the ant subset
            self.ants[index_ant].feature_path.append(next_feature)
            # Remove the chosen feature of the unvisited features
            self.unvisited_feature_indexes = np.delete(
                self.unvisited_feature_indexes,
                np.where(self.unvisited_feature_indexes == next_feature)
            )

            self.redefine_lut(next_feature)

        ant_features_indexes = np.array(self.ants[index_ant].feature_path)
        ant_xs = np.array(self.training_xs[:, ant_features_indexes])

        model = self.model_class()
        model.fit(ant_xs, self.training_ys)
        scores = cross_val_score(model, ant_xs, self.training_ys, cv=5)
        new_ant_acc = scores.mean()

        np.put(self.ant_accuracy_list, index_ant, new_ant_acc)

    def update_pheromones(self):
        """Update the pheromones trail depending on which variant of the algorithm it is selected.
        """

        for selected_feature_index in self.ants[np.argmax(self.ant_accuracy_list)].feature_path:
            sum_delta = self.q / ((1 - np.max(self.ant_accuracy_list)) * 100)

            updated_pheromone = (1 - self.rho) * self.feature_pheromone[selected_feature_index] + sum_delta
            if updated_pheromone < 0.4:
                updated_pheromone = 0.4
            np.put(self.feature_pheromone, selected_feature_index, updated_pheromone)

    def select_features(self):
        self.define_lut()
        for _ in tqdm(range(self.n_iters)):
            self.reset_initial_values()
            for ant_index in range(self.n_ants):
                self.ant_build_subset(ant_index)
            self.update_pheromones()

        return self.evaluate()

    def evaluate(self):
        """Function for printing the entire summary of the algorithm, including the test results.
        """

        selected_features = self.ants[np.argmax(self.ant_accuracy_list)].feature_path
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

        data_training_subset = self.training_xs[:, self.ants[np.argmax(self.ant_accuracy_list)].feature_path]
        data_testing_subset = self.testing_xs[:, self.ants[np.argmax(self.ant_accuracy_list)].feature_path]

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
        # GaussianNB,
        KNeighborsClassifier,  # 30 ants
        DecisionTreeClassifier,  # 30 ants
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
                    n_selected_features=30,
                    model_class=model,
                    n_ants=20,
                    n_iters=30,
                    alpha=1,
                    beta=1,
                    q=3,
                    initial_pheromone=1.0,
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
