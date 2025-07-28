"""
Module pour le chargement et le traitement des données.
Contient les fonctions de lecture de fichiers Excel et de préparation des données.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from text_processing import clean_text

def load_excel_data(file_path, libelle_col="Libellé produit", nature_col="Nature", max_rows=None):
    """
    Charge les données depuis un fichier Excel avec option de limitation du nombre de lignes.
    
    Args:
        file_path (str): Chemin vers le fichier Excel
        libelle_col (str): Nom de la colonne contenant les libellés
        nature_col (str): Nom de la colonne contenant les natures
        max_rows (int, optional): Nombre maximum de lignes à charger (None = toutes)
        
    Returns:
        pd.DataFrame: DataFrame avec les données chargées
    """
    print(f"📁 Chargement du fichier: {file_path}")
    
    if max_rows:
        print(f"⚠️  Mode test: chargement limité à {max_rows} lignes")
    
    # Détection automatique du format et utilisation de l'engine approprié
    if file_path.endswith('.xlsb'):
        try:
            # Essayer d'abord avec calamine (plus rapide pour .xlsb)
            if max_rows:
                df = pd.read_excel(file_path, engine='calamine', nrows=max_rows)
            else:
                df = pd.read_excel(file_path, engine='calamine')
        except:
            # Fallback vers pyxlsb
            if max_rows:
                df = pd.read_excel(file_path, engine='pyxlsb', nrows=max_rows)
            else:
                df = pd.read_excel(file_path, engine='pyxlsb')
    else:
        if max_rows:
            df = pd.read_excel(file_path, nrows=max_rows)
        else:
            df = pd.read_excel(file_path)
    
    # Filtrer pour ne garder que les colonnes nécessaires et supprimer les NaN
    df = df[[libelle_col, nature_col]].dropna()
    
    print(f"✅ Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
    print(f"📊 Catégories uniques: {len(df[nature_col].unique())}")
    
    if max_rows:
        print(f"📋 Chargement en mode test avec {len(df)} échantillons")
    else:
        print("📋 Chargement complet du fichier")
    
    return df

def prepare_data_for_training(df, libelle_col="LIBELLE", nature_col="NATURE", min_samples=30):
    """
    Prépare les données pour l'entraînement du modèle.
    
    Args:
        df (pd.DataFrame): DataFrame avec les données
        libelle_col (str): Nom de la colonne contenant les libellés
        nature_col (str): Nom de la colonne contenant les natures
        min_samples (int): Nombre minimum d'échantillons par catégorie
        
    Returns:
        tuple: (df_filtered, valid_categories, category_counts)
    """
    print("🔍 Analyse des catégories...")
    
    # Calcul du nombre d'échantillons par catégorie
    category_counts = df[nature_col].value_counts()
    
    # Filtrage des catégories avec suffisamment d'échantillons
    valid_categories = category_counts[category_counts >= min_samples].index
    df_filtered = df[df[nature_col].isin(valid_categories)].copy()
    
    print(f"📊 Catégories avec ≥{min_samples} échantillons: {len(valid_categories)}")
    print(f"🎯 Échantillons utilisés: {len(df_filtered)} / {len(df)}")
    print(f"📈 Répartition: min={category_counts[valid_categories].min()}, "
          f"max={category_counts[valid_categories].max()}, "
          f"moyenne={category_counts[valid_categories].mean():.1f}")
    
    return df_filtered, valid_categories, category_counts

def create_tfidf_vectorizers():
    """
    Crée plusieurs configurations de vectoriseurs TF-IDF optimisées pour la mémoire.
    
    Returns:
        list: Liste des configurations TF-IDF (comme dans l'original)
    """
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
    """
    Prépare les features TF-IDF à partir du texte de manière efficace en mémoire.
    
    Args:
        df (pd.DataFrame): DataFrame avec les données
        libelle_col (str): Nom de la colonne contenant les libellés
        tfidf_configs (list): Liste des configurations TF-IDF
        
    Returns:
        np.ndarray: Matrice de features combinées
    """
    print("🔤 Nettoyage du texte...")
    texts_cleaned = df[libelle_col].apply(clean_text)
    
    print("🔢 Création des features TF-IDF...")
    X_combined = []
    
    for i, vectorizer in enumerate(tfidf_configs):
        print(f"   📊 Configuration {i+1}: {vectorizer.analyzer} n-grams {vectorizer.ngram_range}")
        X_tfidf = vectorizer.fit_transform(texts_cleaned).toarray()
        X_combined.append(X_tfidf)
        print(f"      Dimensions: {X_tfidf.shape}")
    
    # Combinaison des features
    X_combined = np.hstack(X_combined)
    print(f"✅ Features combinées: {X_combined.shape}")
    
    return X_combined

def prepare_labels(df, nature_col):
    """
    Prépare les labels encodés pour l'entraînement.
    
    Args:
        df (pd.DataFrame): DataFrame avec les données
        nature_col (str): Nom de la colonne contenant les natures
        
    Returns:
        tuple: (y_encoded, label_encoder)
    """
    print("🏷️  Encodage des labels...")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[nature_col])
    
    print(f"✅ Labels encodés: {len(le.classes_)} classes uniques")
    
    return y_encoded, le
