import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from skcascade.util import search_cascade


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
        self.cascade = search_cascade(X, y, self.scores, lookahead=lookahead, split_criterion=threshold_fit_criterion)
        return self


if __name__ == '__main__':
    model = [18, 1, 1, 1, 0, 0, 4, 0, 1, 0]

    df = pd.read_csv("../data/breastcancer_processed.csv")
    X = df.loc[:, df.columns != 'Benign']
    y = df.Benign

    clf = ScoringCascade(model[1:]).fit(X, y, lookahead=2)
    clf
