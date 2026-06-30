import cv2
import numpy as np
import pandas as pd
import os

def extraire_caracteristiques_image(chemin_image):
    image = cv2.imread(chemin_image)
    
    if image is None:
        return None
    
    total_pixels = image.shape[0] * image.shape[1]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    borne_inferieure_rouille = np.array([0, 50, 50])
    borne_superieure_rouille = np.array([25, 255, 255])

    masque_rouille = cv2.inRange(hsv_image, borne_inferieure_rouille, borne_superieure_rouille)
    nombre_pixels_rouille = cv2.countNonZero(masque_rouille)
    pct_rouille = nombre_pixels_rouille / total_pixels

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    rugosite = np.var(magnitude)

    _, canal_vert, canal_rouge = cv2.split(image)

    moyenne_rouge = np.mean(canal_rouge)
    moyenne_vert = np.mean(canal_vert)

    ratio_rouge_vert = moyenne_rouge / (moyenne_vert + 1e-5)

    return pct_rouille, rugosite, ratio_rouge_vert

def build_dataset_features(dossier_racine):
    categories = {
        'saines': 0,   
        'malades': 1   
    }

    donnees_globales = []

    for nom_dossier, label in categories.items():
        chemin_dossier = os.path.join(dossier_racine, nom_dossier)

        if not os.path.exists(chemin_dossier):
            print(f"Attention : Le dossier {chemin_dossier} n'existe pas.")
            continue

        for nom_fichier in os.listdir(chemin_dossier):
            if nom_fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
                chemin_image = os.path.join(chemin_dossier, nom_fichier)
                 
                features = extraire_caracteristiques_image(chemin_image)

                if features is not None:
                    pct_rouille, rugosite, ratio_rouge_vert = features
                    
                    ma_variable = ratio_rouge_vert

                    donnees_globales.append([nom_fichier, pct_rouille, rugosite, ma_variable, label])

    colonnes = ['Id_Image', 'pct_rouille', 'rugosite', 'ma_variable', 'label_malade']
    df = pd.DataFrame(donnees_globales, columns=colonnes)
    return df