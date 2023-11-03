from itertools import permutations

import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from skcascade.scoring_system import AbstractScoringSystem


class ScoringCascade(BaseEstimator, ClassifierMixin):
    def __init__(self, scores: np.array):
        """

        :param scores: an array of length n (with n being number of features of dataset).
        One scores for each feature, 0 if feature not selected.
        """
        self.scores = scores

        self.cascade = None

    def fit(self, X, y, lookahead=1, probabilistic=False):
        threshold_fit_criterion = "log_loss" if probabilistic else "entropy"
        self.cascade = self._cascadize(X, y, self.scores, lookahead=lookahead, split_criterion=threshold_fit_criterion)
        return self

    @staticmethod
    def _cascadize(X, y, scores, lookahead=1, split_criterion="entropy"):
        optimal_models = [AbstractScoringSystem(criterion=split_criterion).fit(X, y, scores)]
        current_features = set(np.flatnonzero(scores))

        while len(current_features) > 0:
            best_model = None
            best_loss = float('inf')

            # Perform lookahead by considering removing lookahead number of features at a time
            for features_to_remove in permutations(current_features, min(lookahead, len(current_features))):
                # Create a copy of the current set of features and remove the chosen ones
                remaining_features_list = [current_features - set(features_to_remove[:i + 1]) for i in
                                           range(len(features_to_remove))]

                # Train the model with the remaining features
                models = [
                    AbstractScoringSystem(criterion=split_criterion)
                    .fit(X, y, ScoringCascade._mask_scores(scores, remaining_features))
                    for remaining_features in remaining_features_list]

                losses = [model.complexity * (1 - model.score(X, y)) for model in optimal_models + models]
                cascade_loss = -hmean(losses)

                # Update the best model if the current one is better
                if cascade_loss < best_loss:
                    best_model = models[0]
                    best_loss = cascade_loss

            # Add the best model to the list of optimal models
            optimal_models.append(best_model)

            current_features = set(np.flatnonzero(best_model.scores))

        return optimal_models

    def predict_proba(self, X, stage=-1):
        if self.cascade is None:
            raise NotFittedError()
        return self.cascade[stage].predict_proba(X)

    def predict(self, X, stage=-1):
        if self.cascade is None:
            raise NotFittedError()
        return self.cascade[stage].predict(X)

    @staticmethod
    def _mask_scores(scores, features):
        mask = np.zeros_like(scores)
        mask[list(features)] = 1
        return scores * mask


if __name__ == '__main__':
    from pprint import pprint

    scoring_system = np.array([18, 1, 1, 1, 0, 0, 1, 0, 1, 0])

    df = pd.read_csv("../data/breastcancer_processed.csv")
    X_ = df.loc[:, df.columns != 'Benign']
    y_ = df.Benign

    clf = ScoringCascade(scoring_system[1:]).fit(X_, y_, lookahead=1)
    pprint(clf.cascade)
