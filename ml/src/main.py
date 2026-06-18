from part_01 import build_dataset_features
from part_02 import trouver_meilleure_split
import pandas as pd

if __name__ == "__main__":
    # chemin_dataset = "../../dataset" 
    
    # df_features = build_dataset_features(chemin_dataset)
    
    # print("\n--- Extraction Terminée ---")
    # print(f"Nombre total d'images traitées : {len(df_features)}")
    # print("\nAperçu du DataFrame :")
    # print(df_features.head(10))  
    # print(df_features.tail(10))  
    
    # df_features.to_csv("../../dataset/dataset_features.csv", index=False)
    # print("\nFichier 'dataset_features.csv' sauvegardé avec succès !")

    # df = pd.read_csv("../../dataset/dataset_features.csv")
    
    # y_train = df["label_malade"].values

    # variable_a_tester = ["pct_rouille", "rugosite", "ma_variable"]

    # for var in variable_a_tester:
    #     X_col = df[var].values 
        
    #     seuil, purete = trouver_meilleure_split(X_col, y_train)
        
    #     print(f"Variable : {var}")
    #     print(f"  -> Seuil optimal de coupure : {seuil:.4f}")
    #     print(f"  -> Pureté du split obtenue : {purete:.4f}")
    #     print("-" * 40) 

    X_column = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    y = [0,0,0,1,1,1]
    sueil, purete = trouver_meilleure_split(X_column, y)
    print(f"Meilleur sueil: {sueil}, Pureté max : {purete}" )


