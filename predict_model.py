"""
Script de prédiction avec modèle pré-entraîné.
Utilise un modèle sauvegardé pour faire des prédictions sur des données.
"""

import pandas as pd
import numpy as np
from load_model import load_latest_model, predict_with_loaded_model
from data_processing import load_excel_data
from model_utils import print_progress
from sklearn.metrics import accuracy_score, classification_report
import os

# === CONFIGURATION ===
FILE_PATH = "20210614 Ecommerce sales.xlsb"
LIBELLE_COL = "LIBELLE"
NATURE_COL = "NATURE"

def main():
    """Fonction principale de prédiction."""
    
    # === 1. CHARGEMENT DU MODÈLE ===
    print_progress(1, "Chargement du modèle")
    
    try:
        model_components = load_latest_model()
        if model_components is None:
            print("❌ Aucun modèle trouvé. Veuillez d'abord entraîner un modèle.")
            return
        
        print("✅ Modèle chargé avec succès")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return
    
    # === 2. CHARGEMENT DES DONNÉES ===
    print_progress(2, "Chargement des données")
    
    df = load_excel_data(FILE_PATH, LIBELLE_COL, NATURE_COL)
    
    # === 3. PRÉDICTIONS ===
    print_progress(3, "Génération des prédictions")
    
    # Échantillonnage pour test (prendre les 1000 premiers pour exemple)
    df_sample = df.head(1000).copy()
    
    print(f"📊 Prédiction sur {len(df_sample)} échantillons...")
    
    try:
        predictions = predict_with_loaded_model(
            df_sample[LIBELLE_COL].tolist(),
            model_components
        )
        
        df_sample['nature_predite'] = predictions
        
        print("✅ Prédictions générées")
        
        # === 4. COMPARAISON AVEC LES VRAIES VALEURS ===
        if NATURE_COL in df_sample.columns:
            print_progress(4, "Évaluation des prédictions")
            
            # Calcul de la précision
            accuracy = accuracy_score(df_sample[NATURE_COL], df_sample['nature_predite'])
            print(f"🎯 Précision globale: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Analyse des erreurs
            errors = df_sample[df_sample[NATURE_COL] != df_sample['nature_predite']]
            print(f"❌ Erreurs: {len(errors)}/{len(df_sample)} ({len(errors)/len(df_sample)*100:.1f}%)")
            
            if len(errors) > 0:
                print("\n🔍 Exemples d'erreurs:")
                for i, row in errors.head(5).iterrows():
                    print(f"   '{row[LIBELLE_COL][:50]}...'")
                    print(f"      Vraie: {row[NATURE_COL]}")
                    print(f"      Prédite: {row['nature_predite']}")
                    print()
        
        # === 5. EXPORT DES RÉSULTATS ===
        print_progress(5, "Export des résultats")
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"predictions_{timestamp}.xlsx"
        
        df_sample.to_excel(output_file, index=False)
        print(f"📁 Prédictions sauvegardées: {output_file}")
        
    except Exception as e:
        print(f"❌ Erreur lors des prédictions: {e}")
        return
    
    print("🎉 Prédictions terminées avec succès!")

def predict_single_text(text, model_components=None):
    """
    Prédit la nature d'un texte unique.
    
    Args:
        text (str): Texte à classifier
        model_components (dict, optional): Composants du modèle. Si None, charge automatiquement
        
    Returns:
        str: Nature prédite
    """
    if model_components is None:
        model_components = load_latest_model()
        if model_components is None:
            return "Erreur: Aucun modèle disponible"
    
    try:
        predictions = predict_with_loaded_model([text], model_components)
        return predictions[0]
    except Exception as e:
        return f"Erreur: {e}"

def batch_predict_from_file(input_file, output_file=None, text_column="LIBELLE"):
    """
    Effectue des prédictions en lot depuis un fichier.
    
    Args:
        input_file (str): Chemin vers le fichier d'entrée
        output_file (str, optional): Chemin vers le fichier de sortie
        text_column (str): Nom de la colonne contenant le texte
        
    Returns:
        pd.DataFrame: DataFrame avec les prédictions
    """
    print(f"📁 Chargement depuis: {input_file}")
    
    # Chargement du fichier
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        df = pd.read_excel(input_file)
    elif input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        print("❌ Format de fichier non supporté")
        return None
    
    # Chargement du modèle
    model_components = load_latest_model()
    if model_components is None:
        print("❌ Aucun modèle trouvé")
        return None
    
    # Prédictions
    print(f"🔮 Prédiction sur {len(df)} échantillons...")
    predictions = predict_with_loaded_model(df[text_column].tolist(), model_components)
    df['nature_predite'] = predictions
    
    # Sauvegarde
    if output_file is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"batch_predictions_{timestamp}.xlsx"
    
    df.to_excel(output_file, index=False)
    print(f"✅ Prédictions sauvegardées: {output_file}")
    
    return df

if __name__ == "__main__":
    main()
