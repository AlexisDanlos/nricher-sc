"""
Module pour le chargement et le traitement des données.
Contient les fonctions de lecture de fichiers Excel et de préparation des données.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from text_processing import clean_text

def load_excel_data(file_path, libelle_col="LIBELLE", nature_col="NATURE"):
    """
    Charge les données depuis un fichier Excel.
    
    Args:
        file_path (str): Chemin vers le fichier Excel
        libelle_col (str): Nom de la colonne contenant les libellés
        nature_col (str): Nom de la colonne contenant les natures
        
    Returns:
        pd.DataFrame: DataFrame avec les données chargées
    """
    print(f"📁 Chargement du fichier: {file_path}")
    
    # Détection automatique du format
    if file_path.endswith('.xlsb'):
        df = pd.read_excel(file_path, engine='pyxlsb')
    else:
        df = pd.read_excel(file_path)
    
    print(f"✅ Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
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
    Crée plusieurs configurations de vectoriseurs TF-IDF.
    
    Returns:
        dict: Dictionnaire des configurations TF-IDF
    """
    tfidf_configs = {
        'unigrams': TfidfVectorizer(
            max_features=6000,
            ngram_range=(1, 1),
            stop_words=None,
            lowercase=False,
            min_df=2,
            max_df=0.95
        ),
        'bigrams': TfidfVectorizer(
            max_features=6000,
            ngram_range=(2, 2),
            stop_words=None,
            lowercase=False,
            min_df=2,
            max_df=0.95
        ),
        'trigrams': TfidfVectorizer(
            max_features=6000,
            ngram_range=(3, 3),
            stop_words=None,
            lowercase=False,
            min_df=2,
            max_df=0.95
        ),
        'char_ngrams': TfidfVectorizer(
            analyzer='char',
            max_features=6000,
            ngram_range=(2, 4),
            lowercase=False,
            min_df=2,
            max_df=0.95
        )
    }
    
    return tfidf_configs

def prepare_features(df, libelle_col, tfidf_configs):
    """
    Prépare les features TF-IDF à partir du texte.
    
    Args:
        df (pd.DataFrame): DataFrame avec les données
        libelle_col (str): Nom de la colonne contenant les libellés
        tfidf_configs (dict): Configurations TF-IDF
        
    Returns:
        np.ndarray: Matrice de features combinées
    """
    print("🔤 Nettoyage du texte...")
    texts_cleaned = df[libelle_col].apply(clean_text)
    
    print("🔢 Création des features TF-IDF...")
    X_combined = []
    
    for name, vectorizer in tfidf_configs.items():
        print(f"   📊 Configuration {name}...")
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
