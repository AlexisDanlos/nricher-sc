"""
Script d'analyse e-commerce - Enrichissement des données avec dimensions et couleurs.
Ce script se concentre sur l'analyse et l'enrichissement des données existantes.
"""

from data_processing import load_excel_data
from data_analysis import enrich_data_with_features, export_results, analyze_categories, filter_low_frequency_categories, create_summary_report
from model_utils import print_progress

# === CONFIGURATION ===
FILE_PATH = "20210614 Ecommerce sales.xlsb"
LIBELLE_COL = "LIBELLE"
NATURE_COL = "NATURE"

def main():
    """Fonction principale d'analyse et d'enrichissement."""
    
    # === 1. CHARGEMENT DES DONNÉES ===
    print_progress(1, "Chargement des données")
    
    df = load_excel_data(FILE_PATH, LIBELLE_COL, NATURE_COL)
    
    # === 2. ANALYSE DES CATÉGORIES ===
    print_progress(2, "Analyse des catégories")
    
    category_counts = analyze_categories(df, NATURE_COL)
    
    # === 3. FILTRAGE DES CATÉGORIES ===
    print_progress(3, "Filtrage des catégories")
    
    df_filtered, valid_categories, removed_categories = filter_low_frequency_categories(
        df, NATURE_COL, min_samples=30
    )
    
    # === 4. ENRICHISSEMENT DES DONNÉES ===
    print_progress(4, "Enrichissement des données")
    
    df_enriched = enrich_data_with_features(df, LIBELLE_COL)
    
    # === 5. CRÉATION DU RAPPORT ===
    print_progress(5, "Création du rapport de résumé")
    
    report = create_summary_report(df, df_enriched)
    
    # === 6. EXPORT DES RÉSULTATS ===
    print_progress(6, "Export des résultats")
    
    output_file = export_results(df_enriched)
    
    print(f"🎉 Analyse terminée avec succès!")
    print(f"📁 Fichier de sortie: {output_file}")
    print(f"📊 Rapport final: {report}")

if __name__ == "__main__":
    main()
