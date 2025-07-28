"""
Script principal pour l'entraînement du modèle de classification e-commerce.
Ce script se concentre uniquement sur l'entraînement et la sauvegarde du modèle.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import pickle
import os
from datetime import datetime

# Import des modules locaux
from model_utils import TextClassifierNet, print_progress, print_configuration
from data_processing import load_excel_data, prepare_data_for_training, create_tfidf_vectorizers, prepare_features, prepare_labels

# === CONFIGURATION ===
FILE_PATH = "20210614 Ecommerce sales.xlsb"
LIBELLE_COL = "LIBELLE"
NATURE_COL = "NATURE"

# Configuration du modèle
USE_GPU = True
USE_ENSEMBLE = False  # Pour les modèles CPU
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

# Configuration CUDA
if USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"🚀 GPU activé: {torch.cuda.get_device_name(0)}")
    print(f"   Mémoire disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    DEVICE = torch.device("cpu")
    print("🖥️  Mode CPU activé")

def main():
    """Fonction principale d'entraînement."""
    
    # === 1. CHARGEMENT DES DONNÉES ===
    print_progress(1, "Chargement des données")
    
    df = load_excel_data(FILE_PATH, LIBELLE_COL, NATURE_COL)
    
    # === 2. PRÉPARATION DES DONNÉES ===
    print_progress(2, "Préparation des données")
    
    df_filtered, valid_categories, category_counts = prepare_data_for_training(
        df, LIBELLE_COL, NATURE_COL, min_samples=30
    )
    
    # === 3. CRÉATION DES FEATURES ===
    print_progress(3, "Création des features TF-IDF")
    
    tfidf_configs = create_tfidf_vectorizers()
    X = prepare_features(df_filtered, LIBELLE_COL, tfidf_configs)
    y, le_filtered = prepare_labels(df_filtered, NATURE_COL)
    
    # === 4. DIVISION DES DONNÉES ===
    print_progress(4, "Division train/test")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Train: {X_train.shape[0]} échantillons")
    print(f"📊 Test: {X_test.shape[0]} échantillons")
    
    # === 5. ENTRAÎNEMENT DU MODÈLE ===
    print_progress(5, "Entraînement du modèle")
    
    if USE_GPU and torch.cuda.is_available():
        test_score = train_pytorch_model(X_train, X_test, y_train, y_test, tfidf_configs, le_filtered)
    else:
        test_score = train_cpu_model(X_train, X_test, y_train, y_test, le_filtered)
    
    print(f"🎯 Score final: {test_score:.4f}")
    print("🎉 Entraînement terminé avec succès!")

def train_pytorch_model(X_train, X_test, y_train, y_test, tfidf_configs, le_filtered):
    """Entraîne le modèle PyTorch GPU."""
    
    # Configuration du modèle
    input_size = X_train.shape[1]
    hidden_size = 1024
    num_classes = len(le_filtered.classes_)
    
    config = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "device": DEVICE
    }
    print_configuration(config)
    
    # Création du modèle
    model = TextClassifierNet(input_size, hidden_size, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Préparation des données
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
    X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    y_test_tensor = torch.LongTensor(y_test).to(DEVICE)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Entraînement
    print("🔥 Début de l'entraînement...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch_y.size(0)
            correct_predictions += (predicted == batch_y).sum().item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct_predictions / total_samples
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Évaluation
    print("📊 Évaluation du modèle...")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_predicted_cpu = test_predicted.cpu().numpy()
    
    test_score = accuracy_score(y_test, test_predicted_cpu)
    print(f"✅ Précision sur le test: {test_score:.4f}")
    
    # Sauvegarde du modèle
    save_pytorch_model(model, tfidf_configs, le_filtered, test_score, config)
    
    return test_score

def train_cpu_model(X_train, X_test, y_train, y_test, le_filtered):
    """Entraîne le modèle CPU (Random Forest ou Ensemble)."""
    
    if USE_ENSEMBLE:
        print("🌲 Entraînement du modèle ensemble...")
        
        # Création des modèles de base
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            oob_score=True,
            n_jobs=-1
        )
        
        lr = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        
        # Ensemble voting
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('lr', lr)],
            voting='soft',
            n_jobs=-1
        )
        
        ensemble.fit(X_train, y_train)
        pipeline = ensemble
        
    else:
        print("🌲 Entraînement du Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=40,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            oob_score=True,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        pipeline = rf
    
    # Évaluation
    test_predicted = pipeline.predict(X_test)
    test_score = accuracy_score(y_test, test_predicted)
    print(f"✅ Précision sur le test: {test_score:.4f}")
    
    # Sauvegarde
    save_cpu_model(pipeline, le_filtered, test_score)
    
    return test_score

def save_pytorch_model(model, tfidf_configs, le_filtered, test_score, config):
    """Sauvegarde le modèle PyTorch et ses composants."""
    
    print_progress(6, "Sauvegarde du modèle PyTorch")
    
    # Création du dossier
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Génération des noms de fichiers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_filename = f"{model_dir}/pytorch_model_{timestamp}.pth"
    tfidf_filename = f"{model_dir}/tfidf_configs_{timestamp}.pkl"
    label_encoder_filename = f"{model_dir}/label_encoder_{timestamp}.pkl"
    metadata_filename = f"{model_dir}/model_metadata_{timestamp}.pkl"
    
    try:
        # Sauvegarder le modèle PyTorch
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': config,
            'device': DEVICE,
            'test_score': test_score
        }, model_filename, _use_new_zipfile_serialization=False)
        
        # Sauvegarder les configurations TF-IDF
        with open(tfidf_filename, 'wb') as f:
            pickle.dump(tfidf_configs, f)
        
        # Sauvegarder le label encoder
        with open(label_encoder_filename, 'wb') as f:
            pickle.dump(le_filtered, f)
        
        # Sauvegarder les métadonnées
        metadata = {
            'model_type': 'pytorch_gpu',
            'timestamp': timestamp,
            'test_score': test_score,
            'num_classes': config['num_classes'],
            'device': str(DEVICE)
        }
        with open(metadata_filename, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Modèle PyTorch sauvegardé:")
        print(f"   📁 Modèle: {model_filename}")
        print(f"   📁 TF-IDF: {tfidf_filename}")
        print(f"   📁 Encodeur: {label_encoder_filename}")
        print(f"   📁 Métadonnées: {metadata_filename}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")

def save_cpu_model(pipeline, le_filtered, test_score):
    """Sauvegarde le modèle CPU."""
    
    print_progress(6, "Sauvegarde du modèle CPU")
    
    # Création du dossier
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Génération des noms de fichiers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if USE_ENSEMBLE:
        model_filename = f"{model_dir}/ensemble_model_{timestamp}.pkl"
        model_type = 'ensemble_cpu'
    else:
        model_filename = f"{model_dir}/randomforest_model_{timestamp}.pkl"
        model_type = 'randomforest_cpu'
    
    label_encoder_filename = f"{model_dir}/label_encoder_{timestamp}.pkl"
    metadata_filename = f"{model_dir}/model_metadata_{timestamp}.pkl"
    
    try:
        # Sauvegarder le pipeline
        joblib.dump(pipeline, model_filename)
        
        # Sauvegarder le label encoder
        with open(label_encoder_filename, 'wb') as f:
            pickle.dump(le_filtered, f)
        
        # Sauvegarder les métadonnées
        metadata = {
            'model_type': model_type,
            'timestamp': timestamp,
            'test_score': test_score,
            'num_classes': len(le_filtered.classes_),
            'use_ensemble': USE_ENSEMBLE,
            'device': 'cpu'
        }
        with open(metadata_filename, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Modèle CPU sauvegardé:")
        print(f"   📁 Pipeline: {model_filename}")
        print(f"   📁 Encodeur: {label_encoder_filename}")
        print(f"   📁 Métadonnées: {metadata_filename}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")

if __name__ == "__main__":
    main()
