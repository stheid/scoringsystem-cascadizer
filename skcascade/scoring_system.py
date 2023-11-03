import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier


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

    @property
    def complexity(self):
        if self.threshold is None:
            raise NotFittedError()
        return np.count_nonzero(self.scores)
