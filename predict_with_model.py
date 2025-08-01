"""
Script pour utiliser le modèle le plus récent pour prédire la Nature des produits
dans le fichier original 20210614 Ecommerce sales.xlsb
"""

import pandas as pd
import os
import re
from datetime import datetime
import numpy as np
from load_model import ModelLoader
from text_processing import clean_text, extract_colors
from extract_dimensions import extract_dimensions
from color_mapping import color_mapping

def predict_nature_original_file():
    """
    Utilise le modèle le plus récent pour prédire la Nature des produits
    dans le fichier original et compare avec les vraies valeurs
    """
    
    # Fichier à analyser
    input_file = "20210614 Ecommerce sales.xlsb"
    
    print("Prédiction de Nature avec le modèle entraîné")
    print("=" * 60)
    
    # Vérifier que le fichier existe
    if not os.path.exists(input_file):
        print(f"Fichier non trouvé: {input_file}")
        print("Fichiers disponibles:")
        for file in os.listdir("."):
            if file.endswith((".xlsx", ".xlsb")):
                print(f"   - {file}")
        return
    
    # Charger le modèle
    print("Chargement du modèle...")
    loader = ModelLoader()
    
    models = loader.list_available_models()
    if not models:
        print("Aucun modèle trouvé. Exécutez d'abord main.py pour créer un modèle.")
        return
    
    print(f"Modèles disponibles: {len(models)}")
    for i, model in enumerate(models[:3], 1):  # Afficher les 3 plus récents
        print(f"   {i}. {model['timestamp']} - {model['type']} (score: {model['score']:.3f})")
    
    # Charger le modèle le plus récent
    if not loader.load_latest_model():
        print("Impossible de charger le modèle")
        return
    
    model_info = loader.get_model_info()
    print(f"Modèle chargé: {model_info['type']} du {model_info['timestamp']}")
    print(f"   Score d'entraînement: {model_info['score']:.3f}")
    print(f"   Nombre de classes: {model_info['classes']}")
    
    # Charger le fichier original
    print(f"\nChargement du fichier: {input_file}")
    try:
        df = pd.read_excel(input_file)
        print(f"Fichier chargé: {len(df)} lignes")
        
        # Vérifier les colonnes nécessaires
        required_cols = ['Nature', 'Libellé produit']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Colonnes manquantes: {missing_cols}")
            print(f"Colonnes disponibles: {list(df.columns)}")
            return
        
        # Nettoyer les données
        df_clean = df[['Nature', 'Libellé produit']].dropna()
        print(f"Après nettoyage: {len(df_clean)} lignes valides")
        
        # Prédiction par lots pour économiser la mémoire
        batch_size = 5000
        total_rows = len(df_clean)
        predictions = []
        
        print(f"\nPrédiction par lots de {batch_size} éléments...")
        
        for i in range(0, total_rows, batch_size):
            batch_end = min(i + batch_size, total_rows)
            batch_data = df_clean['Libellé produit'].iloc[i:batch_end].tolist()
            
            # Prédiction du lot
            batch_predictions = loader.predict(batch_data)
            predictions.extend(batch_predictions)
            
            # Affichage du progrès
            progress = (batch_end / total_rows) * 100
            print(f"   Lot {i//batch_size + 1}: {progress:.1f}% terminé ({batch_end}/{total_rows})")
        
        # Ajouter les prédictions au DataFrame
        df_clean['predicted_nature'] = predictions
        
        # Extraire les couleurs et dimensions du libellé produit
        print(f"\nExtraction des couleurs et dimensions...")
        df_clean['couleurs_extraites'] = df_clean['Libellé produit'].apply(extract_colors)
        # extract_dimensions now returns (raw, cm)
        df_clean[['dimensions_raw', 'dimensions_cm']] = df_clean['Libellé produit'] \
            .apply(lambda x: pd.Series(extract_dimensions(str(x))))
        
        # Statistiques d'extraction
        colors_found = df_clean['couleurs_extraites'].str.len() > 0
        # consider successful extraction when cm part is not None
        dimensions_found = df_clean['dimensions_cm'].notna()

        print(f"   Couleurs trouvées: {colors_found.sum()}/{len(df_clean)} produits ({(colors_found.sum()/len(df_clean)*100):.1f}%)")
        print(f"   Dimensions trouvées: {dimensions_found.sum()}/{len(df_clean)} produits ({(dimensions_found.sum()/len(df_clean)*100):.1f}%)")

        # Calculer la précision
        correct_predictions = (df_clean['Nature'] == df_clean['predicted_nature']).sum()
        total_predictions = len(df_clean)
        accuracy = (correct_predictions / total_predictions) * 100

        print(f"\nRésultats de la prédiction:")
        print(f"   Précision globale: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
        print(f"   Erreurs: {total_predictions - correct_predictions} prédictions incorrectes")

        # Ajouter une colonne de validation
        df_clean['prediction_correcte'] = df_clean.apply(
            lambda row: 'VRAI' if row['Nature'] == row['predicted_nature'] else 'FAUX', axis=1
        )
        
        # Analyser les erreurs les plus fréquentes
        print(f"\nAnalyse des erreurs:")
        errors = df_clean[df_clean['prediction_correcte'] == 'FAUX']
        
        if len(errors) > 0:
            # Top 10 des erreurs par catégorie réelle
            error_analysis = errors.groupby(['Nature', 'predicted_nature']).size().reset_index(name='count')
            error_analysis = error_analysis.sort_values('count', ascending=False)

            print(f"   Top 10 des confusions les plus fréquentes:")
            for i, row in error_analysis.head(10).iterrows():
                print(f"      {row['count']:4d}x '{row['Nature']}' → '{row['predicted_nature']}'")
            
            # Catégories avec le plus d'erreurs
            print(f"\n   Catégories avec le plus d'erreurs:")
            categories_errors = errors['Nature'].value_counts().head(10)
            for category, error_count in categories_errors.items():
                total_category = len(df_clean[df_clean['Nature'] == category])
                error_rate = (error_count / total_category) * 100
                print(f"      '{category}': {error_count}/{total_category} ({error_rate:.1f}% d'erreur)")
        
        # Analyser les prédictions par catégorie
        print(f"\nPrécision par catégorie (top 15):")
        category_stats = []
        
        for category in df_clean['Nature'].unique():
            category_data = df_clean[df_clean['Nature'] == category]
            category_correct = len(category_data[category_data['prediction_correcte'] == 'VRAI'])
            category_total = len(category_data)
            category_accuracy = (category_correct / category_total) * 100 if category_total > 0 else 0
            
            category_stats.append({
                'category': category,
                'correct': category_correct,
                'total': category_total,
                'accuracy': category_accuracy
            })
        
        # Trier par nombre total (plus représentatif)
        category_stats.sort(key=lambda x: x['total'], reverse=True)

        
        for stat in category_stats[:15]:
            print(f"   {stat['accuracy']:5.1f}% '{stat['category']}' ({stat['correct']}/{stat['total']})")
        
        # Exemples d'erreurs intéressantes
        print(f"\nExemples d'erreurs intéressantes:")
        interesting_errors = errors.sample(min(10, len(errors))) if len(errors) > 0 else pd.DataFrame()
        
        for i, row in interesting_errors.iterrows():
            libelle = row['Libellé produit'][:60] + "..." if len(row['Libellé produit']) > 60 else row['Libellé produit']
            print(f"   • '{libelle}'")
            print(f"     Nature originale: '{row['Nature']}' → Prédite: '{row['predicted_nature']}'")
        
        # Sauvegarder les résultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"predictions_original_file_{timestamp}.xlsx"
        
        # Préparer le DataFrame final avec toutes les colonnes
        df_final = df_clean.copy()
        
        # Ajouter les dimensions et couleurs si disponibles dans le fichier original
        if len(df) == len(df_final):
            # Ajouter toutes les autres colonnes du fichier original
            for col in df.columns:
                if col not in df_final.columns:
                    df_final[col] = df[col].iloc[:len(df_final)]
        
        df_final.to_excel(output_file, index=False)
        print(f"\nRésultats sauvegardés: {output_file}")
        
        # Résumé final
        print(f"\nRésumé final:")
        print(f"   Fichier analysé: {input_file}")
        print(f"   Modèle utilisé: {model_info['type']} ({model_info['timestamp']})")
        print(f"   Précision: {accuracy:.2f}%")
        print(f"   Total produits: {total_predictions}")
        print(f"   Prédictions correctes: {correct_predictions}")
        print(f"   Prédictions incorrectes: {total_predictions - correct_predictions}")
        print(f"   Couleurs extraites: {colors_found.sum()} produits")
        print(f"   Dimensions extraites: {dimensions_found.sum()} produits")
        print(f"   Fichier de sortie: {output_file}")
        
    except Exception as e:
        print(f"Erreur lors du traitement: {str(e)}")
        import traceback
        traceback.print_exc()

def quick_sample_test():
    """Test rapide sur un échantillon de 1000 produits"""
    print("Test rapide sur échantillon")
    print("=" * 40)
    
    # Charger le modèle
    loader = ModelLoader()
    if not loader.load_latest_model():
        print("Impossible de charger le modèle")
        return
    
    # Charger un échantillon
    df = pd.read_excel("20210614 Ecommerce sales.xlsb", nrows=1000)
    df_clean = df[['Nature', 'Libellé produit']].dropna()
    
    print(f"Test sur {len(df_clean)} produits...")
    
    # Prédiction
    predictions = loader.predict(df_clean['Libellé produit'].tolist())
    accuracy = (df_clean['Nature'] == predictions).mean() * 100

    print(f"Précision sur l'échantillon: {accuracy:.2f}%")

if __name__ == "__main__":
    print("Script de prédiction Nature avec modèle entraîné")
    print("=" * 50)
    
    # Demander à l'utilisateur s'il veut un test rapide ou complet
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_sample_test()
    else:
        predict_nature_original_file()
    
    print("=" * 50)
    print("Script terminé")
