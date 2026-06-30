# main.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Importations de vos modules personnalisés
from src.part_01 import build_dataset_features  # Ta fonction d'extraction corrigée
from src.part_02 import trouver_meilleure_split
from src.DecisionTree import build_tree, predict
from src.RandomForest import build_random_forest, predict_random_forest

if __name__ == "__main__":
    # =========================================================================
    # 0. RE-GÉNÉRATION ET MISE À JOUR DU DATASET CSV (PARTIE 1 CORRIGÉE)
    # =========================================================================
    print("="*50)
    print("STAPE 1 : EXTRACTION ET MISE À JOUR DES CARACTÉRISTIQUES")
    print("="*50)
    
    chemin_dataset_images = "../dataset"  # Doit contenir les sous-dossiers /saines et /malades
    chemin_csv = "../dataset/dataset_features.csv"
    
    print("Lancement de l'extraction des caractéristiques (avec ratio Rouge/Vert)...")
    df_nouveau = build_dataset_features(chemin_dataset_images)
    
    # Sauvegarde automatique du nouveau CSV (écrase l'ancien avec les bonnes valeurs)
    os.makedirs(os.path.dirname(chemin_csv), exist_ok=True)
    df_nouveau.to_csv(chemin_csv, index=False)
    print(f"Extraction terminée ! Le fichier '{chemin_csv}' a été mis à jour avec succès.\n")

    # =========================================================================
    # 1. PRÉPARATION ET CHARGEMENT DES DONNÉES
    # =========================================================================
    df = pd.read_csv(chemin_csv)

    # Séparation des caractéristiques (X) et de la cible/label (y)
    X = df[['pct_rouille', 'rugosite', 'ma_variable']].values
    y = df['label_malade'].values

    # Découpage réglementaire : 80% Apprentissage (Train) / 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Données chargées ! Train : {X_train.shape[0]} images, Test : {X_test.shape[0]} images.")

    # =========================================================================
    # 2. ÉVALUATION DE L'ARBRE DE DÉCISION UNIQUE "MAISON"
    # =========================================================================
    print("\n" + "-"*50)
    print("ÉVALUATION DE L'ARBRE DE DÉCISION UNIQUE (MAISON)")
    print("-"*50)
    
    print("Entraînement de l'arbre unique en cours...")
    arbre_maison = build_tree(X_train, y_train, depth=0, maxDepth=3)

    # Prédictions de l'arbre unique
    predictions_train = predict(arbre_maison, X_train)
    predictions_test = predict(arbre_maison, X_test)

    # Calcul des métriques globales pour l'arbre unique
    accuracy_train = accuracy_score(y_train, predictions_train)
    accuracy_test = accuracy_score(y_test, predictions_test)
    prec_tree = precision_score(y_test, predictions_test)
    rec_tree = recall_score(y_test, predictions_test)
    f1_tree = f1_score(y_test, predictions_test)
    cm_tree = confusion_matrix(y_test, predictions_test)

    print(f"Accuracy (Entraînement) : {accuracy_train * 100:.2f}%")
    print(f"Accuracy (Test)         : {accuracy_test * 100:.2f}%")
    print(f"Precision (Test)        : {prec_tree * 100:.2f}%")
    print(f"Recall (Test)           : {rec_tree * 100:.2f}%")
    print(f"F1-Score (Test)         : {f1_tree * 100:.2f}%")
    print("\nMatrice de confusion (Arbre Maison) :")
    print(f"[{cm_tree[0][0]}   {cm_tree[0][1]}]  <- [Vrais Saines (TN)   Fausses Malades (FP)]")
    print(f"[{cm_tree[1][0]}   {cm_tree[1][1]}]  <- [Fausses Saines (FN)  Vrais Malades (TP)]")

    # =========================================================================
    # 3. ÉVALUATION DE LA FORÊT ALÉATOIRE "MAISON" (BAGGING)
    # =========================================================================
    print("\n" + "-"*50)
    print("ÉVALUATION DE LA FORÊT ALÉATOIRE (MAISON)")
    print("-"*50)
    
    print("Entraînement de la Forêt Aléatoire (10 arbres) en cours...")
    foret_maison = build_random_forest(X_train, y_train, n_trees=10, maxDepth=3)

    # Prédictions de la forêt
    predictions_rf_test = predict_random_forest(foret_maison, X_test)

    # Calcul des métriques globales pour la forêt
    acc_rf = accuracy_score(y_test, predictions_rf_test)
    prec_rf = precision_score(y_test, predictions_rf_test)
    rec_rf = recall_score(y_test, predictions_rf_test)
    f1_rf = f1_score(y_test, predictions_rf_test)
    cm_rf = confusion_matrix(y_test, predictions_rf_test)

    print(f"Accuracy (Test)  : {acc_rf * 100:.2f}%")
    print(f"Precision (Test) : {prec_rf * 100:.2f}%")
    print(f"Recall (Test)    : {rec_rf * 100:.2f}%")
    print(f"F1-Score (Test)  : {f1_rf * 100:.2f}%")
    print("\nMatrice de confusion (Forêt Maison) :")
    print(cm_rf)

    # =========================================================================
    # 4. CONFRONTATION ET VALIDATION SCIENTIFIQUE AVEC SCIKIT-LEARN
    # =========================================================================
    print("\n" + "="*50)
    print("      CONFRONTATION AVEC SCIKIT-LEARN")
    print("="*50)

    # Modèle 1 : Arbre de décision standard Sklearn
    arbre_sklearn = DecisionTreeClassifier(max_depth=3, random_state=42)
    arbre_sklearn.fit(X_train, y_train)
    preds_arbre_sk = arbre_sklearn.predict(X_test)
    acc_arbre_sk = accuracy_score(y_test, preds_arbre_sk)

    # Modèle 2 : Forêt Aléatoire standard Sklearn
    foret_sklearn = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    foret_sklearn.fit(X_train, y_train)
    preds_foret_sk = foret_sklearn.predict(X_test)
    acc_foret_sk = accuracy_score(y_test, preds_foret_sk)

    # Tableau final comparatif des exactitudes
    print(f" [Maison]  Arbre Unique Max-Minority : {accuracy_test * 100:.2f}%")
    print(f" [Sklearn] Arbre Unique (Gini/Cart)  : {acc_arbre_sk * 100:.2f}%")
    print("-" * 45)
    print(f" [Maison]  Forêt Aléatoire (Bagging) : {acc_rf * 100:.2f}%")
    print(f" [Sklearn] Forêt Aléatoire Standard  : {acc_foret_sk * 100:.2f}%")
    print("="*50)

    # =========================================================================
    # 5. SAUVEGARDE DES MODÈLES SÉRIALISÉS (.PKL)
    # =========================================================================
    os.makedirs("../modeles_sauvegardes", exist_ok=True)
    
    # Note : Correction apportée ici pour sauvegarder la forêt Sklearn (qui donne la confrontation finale) ou la maison selon tes besoins
    joblib.dump(foret_sklearn, "../modeles_sauvegardes/meilleure_foret.pkl")
    print("[Partie 4] Modèle Forêt Sklearn exporté dans modeles_sauvegardes/meilleure_foret.pkl")
    
    # Sauvegarde de l'arbre fait maison
    joblib.dump(arbre_maison, "../modeles_sauvegardes/meilleur_arbre_maison.pkl")
    print("[Partie 4] Modèle Arbre Maison exporté dans modeles_sauvegardes/meilleur_arbre_maison.pkl")