import numpy as np
from src.DecisionTree import build_tree, predict

def boostrap_sample(X, y):
    n_sample = X.shape[0]

    indices_boostrap = np.random.choice(n_sample, size=n_sample, replace=True)

    return X[indices_boostrap], y[indices_boostrap]

def build_random_forest(X, y, n_trees=10, maxDepth=3):

    forest = []

    for i in range(n_trees):
        X_boot, y_boot = boostrap_sample(X, y)

        arbre_unique = build_tree(X_boot, y_boot, depth=0, maxDepth=maxDepth)

        forest.append(arbre_unique)

    return forest

def predict_random_forest(forest, X_total):
    toutes_predictions = np.array([predict(arbre, X_total) for arbre in forest])

    predictions_finales = []

    n_samples = X_total.shape[0]

    for j in range(n_samples):
        votes_pour_ligne_j = toutes_predictions[:, j]
        vote_gagnant = np.argmax(np.bincount(votes_pour_ligne_j))
        predictions_finales.append(vote_gagnant)

    return np.array(predictions_finales)