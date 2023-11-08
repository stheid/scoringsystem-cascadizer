import numpy as np
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression


class AbstractScoringSystem(BaseEstimator, ClassifierMixin):
    """
    this model only implements partial fitting capabilities. It can only fit the threshold but not the scores
    """

    def __init__(self, criterion="entropy"):
        self.criterion = criterion

        self.scores = None
        self.regressor = None

    def __repr__(self, **kwargs):
        return f"ScoringSystem(scores={self.scores})"

    def fit(self, X, y, scores: np.array):
        """

        :param X: Data
        :param y: labels
        :param scores: full array of scores for each feature. Disabled features must have a score of 0
        """
        self.scores = scores
        self.regressor = IsotonicRegression()
        self.regressor.fit(np.array(X @ scores).reshape(-1, 1), y)
        return self

    def predict_proba(self, X):
        """

        :param X:
        :return: Probability estimate by using sigmoid function and interpreting scores as logits.
        This might not be a good idea if the scores have not been fitted for that purpose initially!
        """
        if self.regressor is None:
            raise NotFittedError()
        proba_true = self.regressor.transform(X @ self.scores)
        proba = np.vstack([1 - proba_true, proba_true]).T
        return proba

    def predict(self, X):
        if self.regressor is None:
            raise NotFittedError()
        return self.predict_proba(X @ self.scores).argmax(axis=1)

    def _expected_entropy(self, X):
        if self.regressor is None:
            raise NotFittedError()
        total_scores, score_freqs = np.unique(X @ self.scores, return_counts=True)
        entropy_values = entropy(self.regressor.transform(total_scores), base=2)
        return np.sum((score_freqs / X.size) * entropy_values)

    @property
    def complexity(self):
        if self.regressor is None:
            raise NotFittedError()
        return np.count_nonzero(self.scores)


if __name__ == '__main__':
    from pprint import pprint
    import pandas as pd

    scoring_system = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0])

    df = pd.read_csv("../data/breastcancer_processed.csv")
    X_ = df.loc[:, df.columns != 'Benign']
    y_ = df.Benign

    clf = AbstractScoringSystem().fit(X_, y_, scores=scoring_system)
    pprint(clf)
    print(clf._expected_entropy(X_))
