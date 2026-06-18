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

    borne_inferieure_vert = np.array([35, 50, 50])
    borne_superieure_vert = np.array([85, 255, 255])

    masque_vert = cv2.inRange(hsv_image, borne_inferieure_vert, borne_superieure_vert)

    nombre_pixels_vert = cv2.countNonZero(masque_vert)
    pct_vert = nombre_pixels_vert / total_pixels

    rugosite = np.var(magnitude)

    return pct_rouille, rugosite, pct_vert

def build_dataset_features(dossier_racine):
    categories = {
        'saines': 0,   
        'malades': 1   
    }

    ma_variable = 0

    donnees_globales = []

    for nom_dossier, label in categories.items():
        chemin_dossier = os.path.join(dossier_racine, nom_dossier)

        for nom_fichier in os.listdir(chemin_dossier):
            if nom_fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
                chemin_image = os.path.join(chemin_dossier, nom_fichier)
                 
                features = extraire_caracteristiques_image(chemin_image)

                pct_rouille, rugosite, pct_vert = features
                ma_variable = pct_vert

                donnees_globales.append([nom_fichier, pct_rouille, rugosite, ma_variable, label])


    colonnes = ['Id_Image', 'pct_rouille', 'rugosite', 'ma_variable', 'label_malade']
    df = pd.DataFrame(donnees_globales, columns=colonnes)
    return df