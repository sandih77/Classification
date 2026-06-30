import streamlit as st

st.set_page_config(
    page_title="Diagnostic du Maïs",
    layout="centered"
)

st.title("Welcome to CornScan")
st.markdown("""
Cette application utilise l'intelligence artificielle (Forêt Aléatoire et Arbre de Décision) 
pour détecter instantanément la **Rouille Polysora** sur les feuilles de maïs.

### Comment utiliser l'application ?
1. Rendez-vous sur la page **Prediction** dans le menu de gauche.
2. Téléchargez une photo de votre feuille de maïs.
3. L'algorithme extrait les caractéristiques et vous donne son diagnostic en temps réel.
4. Consultez l'onglet **Historique** pour revoir vos anciennes analyses.
""")