from itertools import permutations

import numpy as np
from scipy.special import expit
from scipy.stats import hmean
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier


def gen_lookahead(list_, lookahead):
    return permutations(list_, min(lookahead, len(list_)))


def mask_scores(scores, features):
    mask = np.full_like(scores, True).astype(bool)
    mask[list(features)] = False
    s = np.array(scores)
    s[mask] = 0
    return s


def global_loss(losses):
    return -hmean(np.arange(len(losses)) @ (1 - np.array(losses)))


def evaluate_models(models, X, y):
    return global_loss([model.score(X, y) for model in models])


def search_cascade(X, y, scores, lookahead=1, split_criterion="entropy"):
    """

    :param X: Data
    :param y: labels
    :param scores: scores of the original scoringsystem
    :param lookahead: lookahead amount to use for the greedy search
    :param split_criterion: split criterion for the internal decision stump model
    :return:
    """
    optimal_models = []
    current_features = set(np.flatnonzero(scores))

    while len(current_features) > 0:
        best_model = None
        best_performance = -float('inf')

        # Create a list to store the performance of each candidate model
        candidate_performances = []

        # Perform lookahead by considering removing lookahead number of features at a time
        for features_to_remove in gen_lookahead(current_features, lookahead):
            # Create a copy of the current set of features and remove the chosen ones
            remaining_features_list = [current_features - set(features_to_remove[:i + 1]) for i in
                                       range(len(features_to_remove))]

            # Train the model with the remaining features
            models = [
                AbstractScoringSystem(criterion=split_criterion).fit(X, y, mask_scores(scores, remaining_features)) for
                remaining_features in
                remaining_features_list]

            # Evaluate the model using a global performance measure (e.g., cross-validation score)
            cascade_performance = evaluate_models(optimal_models + models, X, y)

            # Store the performance for the candidate model
            candidate_performances.append(cascade_performance)

            # Update the best model if the current one is better
            if cascade_performance > best_performance:
                best_model = models[0]
                current_features = remaining_features_list[0]
                best_performance = cascade_performance

        # Add the best model to the list of optimal models
        optimal_models.append(best_model)

    return optimal_models


class AbstractScoringSystem(BaseEstimator, ClassifierMixin):
    """
    this model only implements partial fitting capabilities. It can only fit the threshold but not the scores
    """

    def __init__(self, criterion="entropy"):
        self.criterion = criterion

        self.scores = None
        self.threshold = None

    def __repr__(self, **kwargs):
        return f"ScoringSystem(scores={self.scores}, threshold={self.threshold})"

    def fit(self, X, y, scores: np.array):
        """

        :param scores: full array of scores for each feature. Disabled features must have a score of 0
        """
        clf = DecisionTreeClassifier(max_depth=1, criterion=self.criterion)
        clf.fit(np.array(X @ scores).reshape(-1, 1), y)
        self.scores = scores
        self.threshold = clf.tree_.threshold[0]
        return self

    def predict_proba(self, X):
        if self.threshold is None:
            raise NotFittedError()
        return expit(self.threshold - X @ self.scores)

    def predict(self, X):
        if self.threshold is None:
            raise NotFittedError()
        return np.array(self.threshold <= X @ self.scores, dtype=int)


if __name__ == '__main__':
    print(mask_scores([1, 2, 1, 0, 0, 3], [1, 2, 5]))
    exit()
    print(global_loss(np.array([.2, .5, 0])))
