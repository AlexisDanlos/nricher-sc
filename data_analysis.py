"""
Module pour l'analyse et l'enrichissement des données e-commerce.
Contient les fonctions d'extraction de dimensions et couleurs, et d'export des résultats.
"""

import pandas as pd
import os
from datetime import datetime
from text_processing import extract_dimensions, extract_colors

def enrich_data_with_features(df, libelle_col="LIBELLE"):
    """
    Enrichit le DataFrame avec l'extraction de dimensions et couleurs.
    
    Args:
        df (pd.DataFrame): DataFrame à enrichir
        libelle_col (str): Nom de la colonne contenant les libellés
        
    Returns:
        pd.DataFrame: DataFrame enrichi avec les nouvelles colonnes
    """
    print("🔍 Extraction des dimensions et couleurs...")
    
    # Extraction des dimensions
    print("   📏 Extraction des dimensions...")
    df["dimension_extraite"] = df[libelle_col].apply(extract_dimensions)
    
    # Extraction des couleurs
    print("   🎨 Extraction des couleurs...")
    df["couleur_extraite"] = df[libelle_col].apply(extract_colors)
    
    # Statistiques d'extraction
    dimensions_found = df["dimension_extraite"].notna().sum()
    colors_found = (df["couleur_extraite"] != "").sum()
    
    print(f"✅ Extraction terminée:")
    print(f"   📏 Dimensions trouvées: {dimensions_found}/{len(df)} ({dimensions_found/len(df)*100:.1f}%)")
    print(f"   🎨 Couleurs trouvées: {colors_found}/{len(df)} ({colors_found/len(df)*100:.1f}%)")
    
    return df

def export_results(df, base_filename="resultats_ecommerce_analyses"):
    """
    Exporte les résultats vers un fichier Excel avec timestamp.
    
    Args:
        df (pd.DataFrame): DataFrame à exporter
        base_filename (str): Nom de base du fichier
        
    Returns:
        str: Nom du fichier créé
    """
    print("💾 Export des résultats...")
    
    # Génération d'un nom de fichier unique avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_filename}_{timestamp}.xlsx"
    
    # Vérification que le fichier n'existe pas déjà (sécurité supplémentaire)
    counter = 1
    base_output = output_filename
    while os.path.exists(output_filename):
        name_part = base_output.replace('.xlsx', '')
        output_filename = f"{name_part}_{counter}.xlsx"
        counter += 1
    
    # Export
    df.to_excel(output_filename, index=False)
    
    print(f"✅ Fichier exporté: {output_filename}")
    print(f"   📊 Lignes: {len(df)}")
    print(f"   📋 Colonnes: {len(df.columns)}")
    
    return output_filename

def analyze_categories(df, nature_col="NATURE"):
    """
    Analyse les catégories dans le DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
        nature_col (str): Nom de la colonne contenant les natures
        
    Returns:
        pd.Series: Comptage des catégories
    """
    print("📊 Analyse des catégories...")
    
    category_counts = df[nature_col].value_counts()
    
    print(f"   📋 Nombre total de catégories: {len(category_counts)}")
    print(f"   📊 Distribution:")
    print(f"      Min: {category_counts.min()} échantillons")
    print(f"      Max: {category_counts.max()} échantillons")
    print(f"      Moyenne: {category_counts.mean():.1f} échantillons")
    print(f"      Médiane: {category_counts.median():.1f} échantillons")
    
    # Top 10 des catégories
    print(f"   🏆 Top 10 des catégories:")
    for i, (category, count) in enumerate(category_counts.head(10).items(), 1):
        print(f"      {i:2d}. {category}: {count} échantillons")
    
    return category_counts

def filter_low_frequency_categories(df, nature_col="NATURE", min_samples=30):
    """
    Filtre les catégories avec peu d'échantillons.
    
    Args:
        df (pd.DataFrame): DataFrame à filtrer
        nature_col (str): Nom de la colonne contenant les natures
        min_samples (int): Nombre minimum d'échantillons par catégorie
        
    Returns:
        tuple: (df_filtered, valid_categories, removed_categories)
    """
    print(f"🔍 Filtrage des catégories avec <{min_samples} échantillons...")
    
    category_counts = df[nature_col].value_counts()
    valid_categories = category_counts[category_counts >= min_samples].index
    removed_categories = category_counts[category_counts < min_samples].index
    
    df_filtered = df[df[nature_col].isin(valid_categories)].copy()
    
    print(f"✅ Filtrage terminé:")
    print(f"   ✅ Catégories conservées: {len(valid_categories)}")
    print(f"   ❌ Catégories supprimées: {len(removed_categories)}")
    print(f"   📊 Échantillons conservés: {len(df_filtered)}/{len(df)} ({len(df_filtered)/len(df)*100:.1f}%)")
    
    return df_filtered, valid_categories, removed_categories

def create_summary_report(df, enriched_df=None):
    """
    Crée un rapport de résumé des analyses.
    
    Args:
        df (pd.DataFrame): DataFrame original
        enriched_df (pd.DataFrame, optional): DataFrame enrichi
        
    Returns:
        dict: Dictionnaire contenant les statistiques
    """
    print("📋 Création du rapport de résumé...")
    
    report = {
        'total_products': len(df),
        'total_categories': df['NATURE'].nunique() if 'NATURE' in df.columns else 0,
        'creation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if enriched_df is not None:
        if 'dimension_extraite' in enriched_df.columns:
            report['dimensions_extracted'] = enriched_df['dimension_extraite'].notna().sum()
            report['dimension_extraction_rate'] = report['dimensions_extracted'] / len(enriched_df) * 100
        
        if 'couleur_extraite' in enriched_df.columns:
            report['colors_extracted'] = (enriched_df['couleur_extraite'] != "").sum()
            report['color_extraction_rate'] = report['colors_extracted'] / len(enriched_df) * 100
    
    print("✅ Rapport créé:")
    for key, value in report.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    return report
