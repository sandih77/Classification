import numpy as np

def calculer_purete_noeud(y_noeud):
    N = len(y_noeud)
    
    if N == 0:
        return 0.0
    
    nb_saines = np.sum(y_noeud == 0)
    nb_malades = np.sum(y_noeud == 1)

    purete = max(nb_saines / N, nb_malades / N)
    return purete

def trouver_meilleure_split(X_column, y):
    X_column = np.array(X_column)
    y = np.array(y)

    N_total = len(y)

    donnee_triee = np.sort(np.unique(X_column))

    meilleur_seuil = None
    purete_max = -1

    for i in range (len(donnee_triee) - 1):
        milieu = (donnee_triee[i] + donnee_triee[i + 1]) / 2
        
        y_gauche = y[X_column <= milieu]
        y_droite = y[X_column > milieu]

        p_gauche = calculer_purete_noeud(y_gauche)
        p_droite = calculer_purete_noeud(y_droite)

        N_gauche = len(y_gauche)
        N_droite = len(y_droite)

        p_split = ((N_gauche / N_total) * p_gauche) + ((N_droite / N_total) * p_droite) 

        if p_split > purete_max:
            purete_max = p_split
            meilleur_seuil = milieu

    return meilleur_seuil, purete_max 