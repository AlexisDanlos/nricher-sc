"""
Module pour le chargement et le traitement des données.
Contient les fonctions de lecture de fichiers Excel et de préparation des données.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from text_processing import clean_text

def load_excel_data(file_path: str, libelle_col="Libellé produit", nature_col="Nature", max_rows=None):
    print(f"Chargement du fichier: {file_path}")
    
    if max_rows:
        print(f"Mode test: chargement limité à {max_rows} lignes")
    
    # Détection automatique du format et utilisation de l'engine approprié
    if file_path.endswith('.xlsb') or file_path.endswith('.xlsx'):
        try:
            # Essayer d'abord avec calamine (plus rapide pour .xlsb)
            if max_rows:
                df = pd.read_excel(file_path, nrows=max_rows)
            else:
                df = pd.read_excel(file_path)
        except:
            # Fallback vers pyxlsb
            if max_rows:
                df = pd.read_excel(file_path, nrows=max_rows)
            else:
                df = pd.read_excel(file_path)
    else:
        if max_rows:
            df = pd.read_excel(file_path, nrows=max_rows)
        else:
            df = pd.read_excel(file_path)
    
    # Filtrer pour ne garder que les colonnes nécessaires et supprimer les NaN
    df = df[[libelle_col, nature_col]].dropna()
    
    print(f"Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
    print(f"Catégories uniques: {len(df[nature_col].unique())}")
    
    if max_rows:
        print(f"Chargement en mode test avec {len(df)} échantillons")
    else:
        print("Chargement complet du fichier")
    
    return df

def augment_rare_categories(df, nature_col, min_samples_per_category):
    """
    Augmente les données en dupliquant les échantillons des catégories rares
    jusqu'à atteindre min_samples_per_category échantillons par catégorie.
    """
    print(f"Augmentation des catégories avec <{min_samples_per_category} échantillons...")
    
    category_counts = df[nature_col].value_counts()
    rare_categories = category_counts[category_counts < min_samples_per_category].index
    
    if len(rare_categories) == 0:
        print("Aucune catégorie rare détectée, pas d'augmentation nécessaire")
        return df.copy()
    
    print(f"Catégories à augmenter: {len(rare_categories)}")
    
    # Liste pour stocker tous les DataFrames
    augmented_dfs = [df.copy()]
    
    # Pour chaque catégorie rare, dupliquer jusqu'à atteindre min_samples_per_category
    for category in rare_categories:
        current_count = category_counts[category]
        needed_samples = min_samples_per_category - current_count
        
        if needed_samples <= 0:
            continue
            
        # Récupérer tous les échantillons de cette catégorie
        category_samples = df[df[nature_col] == category].copy()
        
        # Calculer combien de fois dupliquer et combien d'échantillons supplémentaires
        full_duplications = needed_samples // current_count
        remaining_samples = needed_samples % current_count
        
        print(f"   {category}: {current_count} → {min_samples_per_category} "
              f"(+{needed_samples} échantillons)")
        
        # Dupliquer complètement les données autant de fois que nécessaire
        for _ in range(full_duplications):
            augmented_dfs.append(category_samples.copy())
        
        # Ajouter les échantillons restants si nécessaire
        if remaining_samples > 0:
            # Échantillonner aléatoirement les échantillons restants
            remaining_df = category_samples.sample(n=remaining_samples, random_state=42).copy()
            augmented_dfs.append(remaining_df)
    
    # Combiner tous les DataFrames
    df_augmented = pd.concat(augmented_dfs, ignore_index=True)
    
    # Mélanger les données pour éviter que les duplicatas soient groupés
    df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Augmentation terminée: {len(df)} → {len(df_augmented)} échantillons")
    
    return df_augmented

def prepare_data_for_training(df, libelle_col="LIBELLE", nature_col="NATURE", min_samples=30):
    print("Analyse des catégories...")
    
    # Calcul du nombre d'échantillons par catégorie
    category_counts = df[nature_col].value_counts()
    
    # Filtrage des catégories avec suffisamment d'échantillons
    valid_categories = category_counts[category_counts >= min_samples].index
    df_filtered = df[df[nature_col].isin(valid_categories)].copy()
    
    print(f"Catégories avec ≥{min_samples} échantillons: {len(valid_categories)}")
    print(f"Échantillons utilisés: {len(df_filtered)} / {len(df)}")
    print(f"Répartition: min={category_counts[valid_categories].min()}, "
          f"max={category_counts[valid_categories].max()}, "
          f"moyenne={category_counts[valid_categories].mean():.1f}")
    
    return df_filtered, valid_categories, category_counts

def create_tfidf_vectorizers():
    from numpy import float32
    
    tfidf_configs = [
        # Configuration 1: Features générales (réduit pour éviter OOM)
        TfidfVectorizer(
            max_features=4000,  # Réduit pour éviter OOM
            ngram_range=(1, 2),  # Réduit les n-grams
            min_df=3,
            max_df=0.9,
            dtype=float32,
            sublinear_tf=True,
            analyzer='word',
            token_pattern=r'\b[a-zA-Z]{2,}\b',
            stop_words=None,
            use_idf=True,
            smooth_idf=True,
            norm='l2',
            lowercase=False
        ),
        # Configuration 2: Focus sur les caractères (réduit pour éviter OOM)
        TfidfVectorizer(
            max_features=2000,  # Réduit pour éviter OOM
            ngram_range=(2, 3),  # Réduit les n-grams
            min_df=5,
            max_df=0.85,
            dtype=float32,
            sublinear_tf=True,
            analyzer='char',
            stop_words=None,
            use_idf=True,
            smooth_idf=True,
            norm='l2',
            lowercase=False
        )
    ]
    
    return tfidf_configs

def prepare_features(df, libelle_col, tfidf_configs):
    print("Nettoyage du texte...")
    texts_cleaned = df[libelle_col].apply(clean_text)
    
    print("Création des features TF-IDF...")
    X_combined = []
    
    for i, vectorizer in enumerate(tfidf_configs):
        print(f"   Configuration {i+1}: {vectorizer.analyzer} n-grams {vectorizer.ngram_range}")
        X_tfidf = vectorizer.fit_transform(texts_cleaned).toarray()
        X_combined.append(X_tfidf)
        print(f"   Dimensions: {X_tfidf.shape}")
    
    # Combinaison des features
    X_combined = np.hstack(X_combined)
    print(f"✅ Features combinées: {X_combined.shape}")
    
    return X_combined

def prepare_labels(df, nature_col):
    print("Encodage des labels...")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[nature_col])
    
    print(f"Labels encodés: {len(le.classes_)} classes uniques")
    
    return y_encoded, le
