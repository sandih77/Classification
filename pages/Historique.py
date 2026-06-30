# pages/2_Historique.py
import streamlit as st
import os
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFICATION_DIR = os.path.dirname(CURRENT_DIR)
UPLOADS_DIR = os.path.join(CLASSIFICATION_DIR, "uploads")
chemin_csv_historique = os.path.join(UPLOADS_DIR, "historique.csv")

st.title("Historique des Diagnostics")
st.markdown("Retrouvez l'ensemble des requêtes et analyses effectuées par l'arbre de décision.")

if not os.path.exists(chemin_csv_historique) or os.stat(chemin_csv_historique).st_size == 0:
    st.info("Aucun historique disponible. Effectuez votre premier diagnostic dans l'onglet 'Prediction' !")
else:
    df_historique = pd.read_csv(chemin_csv_historique)
    
    df_historique = df_historique.iloc[::-1].reset_index(drop=True)
    
    total_scans = len(df_historique)
    malades = len(df_historique[df_historique["Verdict"] == "MALADE"])
    saines = total_scans - malades
    taux_infection = (malades / total_scans) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Analyses", total_scans)
    col2.metric("Cas Malades", malades, delta=f"{taux_infection:.1f}% du total", delta_color="inverse")
    col3.metric("Feuilles Saines", saines)
    
    st.write("---")
    
    st.subheader("Registre complet des scans")
    
    def colorier_verdict(val):
        color = '#ffcccc' if val == "MALADE" else '#ccffcc'
        return f'background-color: {color}; color: black; font-weight: bold;'
        
    df_style = df_historique.style.map(colorier_verdict, subset=['Verdict'])
    st.dataframe(df_style, use_container_width=True)
    
    st.write("---")
    
    st.subheader("Visionneuse d'images de l'historique")
    image_selectionnee = st.selectbox(
        "Sélectionnez le fichier d'une ancienne analyse pour revoir la photo :", 
        df_historique["Image"].unique()
    )
    
    if image_selectionnee:
        chemin_image_historique = os.path.join(UPLOADS_DIR, image_selectionnee)
        
        if os.path.exists(chemin_image_historique):
            infos_img = df_historique[df_historique["Image"] == image_selectionnee].iloc[0]
            
            col_img, col_txt = st.columns([1, 1])
            with col_img:
                st.image(chemin_image_historique, caption=f"Rappel photo : {image_selectionnee}", use_container_width=True)
            with col_txt:
                st.markdown(f"**Date du scan :** `{infos_img['Date']}`")
                if infos_img['Verdict'] == "MALADE":
                    st.error(f"Verdict d'époque : {infos_img['Verdict']}")
                else:
                    st.success(f"Verdict d'époque : {infos_img['Verdict']}")
                st.write(f"- **Pourcentage de rouille :** {infos_img['Pct_Rouille_Pct']}%")
                st.write(f"- **Rugosité numérique :** {infos_img['Rugosite']}")
        else:
            st.caption("Le fichier image d'origine n'est plus présent dans le dossier 'uploads/', mais ses données d'extraction restent sauvegardées dans le tableau ci-dessus.")