import numpy as np
from src.model.Node import Node
from src.part_02 import trouver_meilleure_split

def build_tree(X, y, depth=0, maxDepth=5):
    num_samples, num_features = X.shape

    if depth >= maxDepth or len(np.unique(y)) == 1 or num_samples < 2:
        classe_majoritaire = int(np.round(np.mean(y)))
        return Node(value = classe_majoritaire) 

    meilleure_purete = -1
    meilleur_feature = None
    meilleur_seuil = None

    for i in range (num_features):
        X_column = X[:, i]
        seuil, purete = trouver_meilleure_split(X_column, y)

        if seuil is not None and purete > meilleure_purete:
            meilleure_purete = purete
            meilleur_feature = i
            meilleur_seuil = seuil
        
    if meilleur_seuil is None:
        classe_majoritaire = int(np.round(np.mean(y)))
        return Node(value = classe_majoritaire)

    indices_gauches =  X[:, meilleur_feature] <= meilleur_seuil
    indices_droites = X[:, meilleur_feature] > meilleur_seuil

    branche_gauche = build_tree(X[indices_gauches], y[indices_gauches], depth + 1, maxDepth)
    branche_droite = build_tree(X[indices_droites], y[indices_droites], depth + 1, maxDepth)

    return Node(feature=meilleur_feature, threshold=meilleur_seuil, left=branche_gauche, right=branche_droite) 

def predire_ligne(noeud, x_ligne):
    if noeud.is_leaf():
        return noeud.value
    
    if x_ligne[noeud.feature] <= noeud.threshold:
        return predire_ligne(noeud.left, x_ligne)

    else:
        return predire_ligne(noeud.right, x_ligne)
    
def predict(noeud_racine, X_total):
    return np.array([predire_ligne(noeud_racine, ligne) for ligne in X_total])