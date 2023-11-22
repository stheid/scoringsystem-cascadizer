from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from skcascade import AbstractScoringSystem

if __name__ == '__main__':
    dataset = "thorax"
    df = pd.read_csv(f"../data/{dataset}.csv")
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    pos = dict()
    G = nx.DiGraph()
    scoreset = [0, 1]
    for scores in tqdm(product(scoreset, repeat=X.shape[1]), total=len(scoreset) ** X.shape[1]):
        # performance = accuracy_score(y_test, AbstractScoringSystem(scores).fit(X_train, y_train).predict(X_test))
        performance = AbstractScoringSystem(scores).fit(X_train, y_train)._expected_entropy(X_test)
        G.add_node(scores)
        pos[scores] = np.count_nonzero(scores), performance
        for index in np.nonzero(scores)[0]:
            from_ = np.array(scores)
            from_[index] = 0
            G.add_edge(tuple(from_), scores)

    f = plt.figure(figsize=(10, 7))
    ax = f.add_subplot(111)
    nx.draw_networkx_nodes(G, pos, node_size=100, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)
    # highlight cascade
    plt.box(False)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlim(-0.5, X.shape[1] + .5)

    f.savefig(f"{dataset}.png")
