"""
Script principal pour l'entraînement du modèle de classification e-commerce.
Ce script se concentre uniquement sur l'entraînement et la sauvegarde du modèle.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os
from datetime import datetime

# Import des modules locaux
from model_utils import TextClassifierNet, print_progress
from data_processing import load_excel_data, create_tfidf_vectorizers, prepare_features, prepare_labels

# === CONFIGURATION ===
FILE_PATH = "ecommerce_corrected_20250729_184443.xlsx"
LIBELLE_COL = "Libellé produit"
NATURE_COL = "Nature"

# Configuration des données
LIMIT_ROWS = 10000  # Limiter pour les tests (ex: 20000), None pour tout charger

# Configuration du modèle
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

# Configuration CUDA - PyTorch peut utiliser CPU ou GPU automatiquement
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"🚀 GPU activé: {torch.cuda.get_device_name(0)}")
    print(f"   Mémoire disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    DEVICE = torch.device("cpu")
    print("🖥️  Mode CPU activé (PyTorch)")

def main():
    """Fonction principale d'entraînement."""
    
    # === 1. CHARGEMENT DES DONNÉES ===
    print_progress(1, "Chargement des données")
    
    df = load_excel_data(FILE_PATH, LIBELLE_COL, NATURE_COL, max_rows=LIMIT_ROWS)
    
    # === 2. PRÉPARATION DES DONNÉES ===
    print_progress(2, "Préparation des données")
    
    # Filtrage des catégories rares pour améliorer les performances
    # Utilisation de min=2 pour permettre la stratification (train_test_split nécessite ≥2 échantillons/classe)
    min_samples_per_category = 1
    category_counts = df[NATURE_COL].value_counts()
    valid_categories = category_counts[category_counts >= min_samples_per_category].index
    
    # Filtration du dataset pour ne garder que les catégories avec assez d'exemples
    df_filtered = df[df[NATURE_COL].isin(valid_categories)].copy()
    removed_count = len(df) - len(df_filtered)
    
    # Reset des indices pour assurer la continuité
    df_filtered = df_filtered.reset_index(drop=True)
    
    if removed_count > 0:
        print(f"⚠️  {removed_count} produits supprimés (catégories rares avec <{min_samples_per_category} exemples)")
        print(f"📊 Dataset d'entraînement: {len(df_filtered)} produits, {len(valid_categories)} catégories")
    
    print(f"📊 Catégories avec ≥{min_samples_per_category} échantillons: {len(valid_categories)}")
    print(f"🎯 Échantillons utilisés: {len(df_filtered)} / {len(df)}")
    print(f"📈 Répartition: min={category_counts[valid_categories].min()}, max={category_counts[valid_categories].max()}, moyenne={category_counts[valid_categories].mean():.1f}")
    
    # === 3. CRÉATION DES FEATURES ===
    print_progress(3, "Création des features TF-IDF")
    
    tfidf_configs = create_tfidf_vectorizers()
    X = prepare_features(df_filtered, LIBELLE_COL, tfidf_configs)
    y, le_filtered = prepare_labels(df_filtered, NATURE_COL)
    
    # === 4. DIVISION DES DONNÉES ===
    print_progress(4, "Division train/test")
    
    # Vérification que toutes les classes ont au moins 2 échantillons pour la stratification
    unique, counts = np.unique(y, return_counts=True)
    min_class_count = counts.min()
    if min_class_count < 2:
        print(f"⚠️  Stratification désactivée: certaines classes ont <2 échantillons (min={min_class_count})")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    print(f"📊 Train: {X_train.shape[0]} échantillons")
    print(f"📊 Test: {X_test.shape[0]} échantillons")
    
    # === 5. ENTRAÎNEMENT DU MODÈLE ===
    print_progress(5, "Entraînement du modèle PyTorch")
    
    test_score = train_pytorch_model(X_train, X_test, y_train, y_test, tfidf_configs, le_filtered, valid_categories)
    
    print(f"🎯 Score final: {test_score:.4f}")
    print("🎉 Entraînement terminé avec succès!")

def train_pytorch_model(X_train, X_test, y_train, y_test, tfidf_configs, le_filtered, valid_categories):
    """Entraîne le modèle PyTorch - Compatible CPU/GPU."""
    
    # Configuration du modèle adaptatif selon le device
    input_size = X_train.shape[1]
    
    # Adapter l'architecture selon le device et la taille des données
    if DEVICE.type == 'cuda':
        hidden_size = min(1024, input_size // 4)  # Architecture GPU
        batch_size = 64
        print(f"🔥 Entraînement modèle PyTorch sur GPU avec {len(le_filtered.classes_)} catégories...")
    else:
        hidden_size = min(512, input_size // 6)   # Architecture CPU plus légère
        batch_size = 32
        print(f"🖥️  Entraînement modèle PyTorch sur CPU avec {len(le_filtered.classes_)} catégories...")
        
    num_classes = len(le_filtered.classes_)
    
    config = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "device": DEVICE,
        "dropout_rate": 0.4
    }
    
    print(f"📊 Architecture: {input_size} → {hidden_size} → {hidden_size//2} → {hidden_size//4} → {num_classes}")
    print(f"⚙️  Paramètres: Batch={batch_size}, LR={LEARNING_RATE}, Dropout=0.4, Label Smoothing=0.1")
    print(f"🎯 Device: {DEVICE}, Validation tous les 5 époques")
    print("-" * 60)
    
    model = TextClassifierNet(
        input_size, 
        hidden_size, 
        num_classes, 
        dropout_rate=0.4
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Créer les datasets avec la bonne taille de batch
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).to(DEVICE),
        torch.LongTensor(y_train).to(DEVICE)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Préparer les tenseurs de test
    X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    y_test_tensor = torch.LongTensor(y_test).to(DEVICE)
        
    # Entraînement avancé avec un seul modèle optimisé
    model.train()
    best_accuracy = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):  # Utiliser la variable globale EPOCHS
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Mixup data augmentation occasionnel
            if np.random.random() < 0.1 and len(data) > 1:
                lam = np.random.beta(0.2, 0.2)
                index = torch.randperm(data.size(0)).to(DEVICE)
                mixed_data = lam * data + (1 - lam) * data[index]
                output = model(mixed_data)
                loss = lam * criterion(output, target) + (1 - lam) * criterion(output, target[index])
            else:
                output = model(data)
                loss = criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            
            # Affichage du progrès par batch pour les premières époques ou époques critiques
            if (epoch < 10 or epoch % 20 == 0) and batch_idx % (len(train_loader) // 4) == 0:
                progress_pct = (batch_idx + 1) / len(train_loader) * 100
                current_avg_loss = total_loss / (batch_idx + 1)
                print(f"     Batch {batch_idx+1:3d}/{len(train_loader)} ({progress_pct:5.1f}%) - Loss: {current_avg_loss:.4f}")
        
        scheduler.step()
        
        # Validation et affichage du progrès plus fréquent
        if epoch % 5 == 0 or epoch >= 65:  # Validation tous les 5 époques
            model.eval()
            with torch.no_grad():
                test_output = model(X_test_tensor)
                _, predicted = torch.max(test_output.data, 1)
                accuracy = (predicted == y_test_tensor).float().mean().item()
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0
                    # Sauvegarder le meilleur modèle
                    best_model_state = model.state_dict().copy()
                    improvement_indicator = "⬆️"
                else:
                    patience_counter += 1
                    improvement_indicator = "➡️"
                
                # Calcul du learning rate actuel
                current_lr = optimizer.param_groups[0]['lr']
                
                # Affichage détaillé du progrès
                avg_loss = total_loss / batch_count
                print(f"   {improvement_indicator} Époque {epoch+1:3d}: Loss={avg_loss:.4f}, Acc={accuracy:.3f}, Best={best_accuracy:.3f}, LR={current_lr:.6f}, Patience={patience_counter}")
                
                if patience_counter >= 3 and epoch >= 45:  # Early stopping plus patient
                    print(f"   🛑 Early stopping à l'époque {epoch+1} (patience épuisée)")
                    break
                    
            model.train()
        else:
            # Affichage rapide de la loss pour les autres époques
            avg_loss = total_loss / batch_count
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   📈 Époque {epoch+1:3d}: Loss={avg_loss:.4f}, LR={current_lr:.6f}")
    
    # Restaurer le meilleur modèle
    if best_model_state:
        model.load_state_dict(best_model_state)
    print(f"   ✅ Modèle terminé - Meilleure précision: {best_accuracy:.3f}")
    
    # Wrapper simplifié pour un seul modèle
    class SingleModelWrapper:
        def __init__(self, model, device):
            self.model = model
            self.device = device
            
        def predict(self, X):
            # X est déjà transformé et combiné
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                output = self.model(X_tensor)
                _, predicted = torch.max(output.data, 1)
                return predicted.cpu().numpy()
    
    # Créer le wrapper
    pipeline = SingleModelWrapper(model, DEVICE)
    
    # Score final avec PyTorch
    final_predictions = pipeline.predict(X_test)
    test_score = (final_predictions == y_test).mean()  # Calculer l'accuracy directement
    
    # Sauvegarde du modèle
    save_pytorch_model(model, tfidf_configs, le_filtered, test_score, config, valid_categories, X_train, X_test)
    
    return test_score

def save_pytorch_model(model, tfidf_configs, le_filtered, test_score, config, valid_categories, X_train, X_test):
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
            'model_type': 'pytorch',
            'timestamp': timestamp,
            'test_score': test_score,
            'num_classes': config['num_classes'],
            'valid_categories': valid_categories.tolist(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'device': str(DEVICE),
            'architecture': f"{config['input_size']} → {config['hidden_size']} → {config['num_classes']}"
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

if __name__ == "__main__":
    main()
