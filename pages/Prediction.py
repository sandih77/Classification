# pages/1_Prediction.py
import streamlit as st
import os
import sys
import joblib
import numpy as np
import pandas as pd
import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # Dossier pages/
CLASSIFICATION_DIR = os.path.dirname(CURRENT_DIR)       # Dossier Classification/
ML_DIR = os.path.join(CLASSIFICATION_DIR, "ml")          # Dossier ml/
SRC_DIR = os.path.join(ML_DIR, "src")                    # Dossier ml/src/

for dossier in [CLASSIFICATION_DIR, ML_DIR, SRC_DIR]:
    if dossier not in sys.path:
        sys.path.insert(0, dossier)

from ml.src.part_01 import extraire_caracteristiques_image  
from ml.src.DecisionTree import predict

st.title("Diagnostic en temps réel")
st.markdown("Téléchargez une photo de feuille de maïs pour détecter la présence de Rouille Polysora.")

fichier_image = st.file_uploader("Choisissez une image de feuille...", type=["jpg", "jpeg", "png"])

if fichier_image is not None:
    uploads_dir = os.path.join(CLASSIFICATION_DIR, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    
    chemin_image = os.path.join(uploads_dir, fichier_image.name)
    with open(chemin_image, "wb") as f:
        f.write(fichier_image.getbuffer())

    st.image(chemin_image, caption=f"Image : {fichier_image.name}", use_container_width=True)
    
    with st.spinner("Analyse et extraction des caractéristiques en cours..."):
        try:
            features = extraire_caracteristiques_image(chemin_image)
            
            chemin_modele = os.path.join(CLASSIFICATION_DIR, "modeles_sauvegardes", "meilleur_arbre_maison.pkl")
            if not os.path.exists(chemin_modele):
                st.error(f"Fichier modèle introuvable à l'emplacement : {chemin_modele}. Veuillez ré-exécuter ml/main.py")
                st.stop()
                
            modele = joblib.load(chemin_modele)
            
            prediction = predict(modele, np.asarray([features]))[0]
            
            chemin_csv_historique = os.path.join(uploads_dir, "historique.csv")
            verdict_texte = "MALADE" if prediction == 1 else "SAINE"
            
            nouvelle_entree = pd.DataFrame([{
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Image": fichier_image.name,
                "Pct_Rouille_Pct": round(features[0] * 100, 2),
                "Rugosite": round(features[1], 2),
                "Verdict": verdict_texte
            }])
            
            if os.path.exists(chemin_csv_historique):
                nouvelle_entree.to_csv(chemin_csv_historique, mode='a', header=False, index=False)
            else:
                nouvelle_entree.to_csv(chemin_csv_historique, mode='w', header=True, index=False)

            st.success("Analyse terminée et sauvegardée dans l'historique !")
            
            st.subheader("Verdict de l'Arbre Max-Minority :")
            if prediction == 1:
                st.error("FEUILLE MALADE (Présence de Rouille Polysora)")
            else:
                st.success("FEUILLE SAINE (En bonne santé)")
                
            st.write("---")
            st.write(f"**Détails techniques extraits :**")
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Pourcentage de rouille", f"{features[0]*100:.2f}%")
            col_m2.metric("Rugosité mesurée", f"{features[1]:.2f}")
            
        except Exception as e:
            st.error(f"Une erreur est survenue lors de l'extraction ou de la prédiction : {e}")