"""
Script pour charger et utiliser les modèles sauvegardés par main.py
Permet de faire des prédictions sans refaire l'entraînement
"""

import os
import pickle
import joblib
import pandas as pd
import re
from datetime import datetime
import numpy as np

# Configuration GPU/CPU selon disponibilité
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch non disponible - Chargement modèles CPU uniquement")

class TextClassifierNet(nn.Module):
    """Même architecture que dans main.py - nécessaire pour charger le modèle PyTorch"""
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.4):
        super(TextClassifierNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc5 = nn.Linear(hidden_size // 4, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size // 2)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size // 4)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.elu = nn.ELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        residual = x
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = x + residual
        
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.elu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = self.batch_norm4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc5(x)
        return x

def clean_text(text):
    """Même fonction de nettoyage que dans main.py"""
    text = str(text).lower()
    
    text = re.sub(r'&', ' et ', text)
    text = re.sub(r'%', ' pourcent ', text)
    text = re.sub(r'\+', ' plus ', text)
    text = re.sub(r'@', ' arobase ', text)
    
    dimension_patterns = re.findall(r'\d+(?:[,\.]\d+)?\s*[xX×*]\s*\d+(?:[,\.]\d+)?(?:\s*[xX×*]\s*\d+(?:[,\.]\d+)?)?', text)
    for i, dim in enumerate(dimension_patterns):
        text = text.replace(dim, f' dimension{i} ')
    
    text = re.sub(r'\b(xs|s|m|l|xl|xxl|xxxl)\b', r' taille_\1 ', text)
    text = re.sub(r'\b(\d+)\s*(cm|mm|m|kg|g|ml|l)\b', r' mesure_\1_\2 ', text)
    
    text = re.sub(r'[^\w\s]', ' ', text)
    
    words = text.split()
    words = [word for word in words if len(word) >= 2]
    
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class ModelLoader:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.model = None
        self.label_encoder = None
        self.metadata = None
        self.tfidf_configs = None
        self.pipeline = None
        
    def list_available_models(self):
        """Liste tous les modèles disponibles dans le dossier"""
        if not os.path.exists(self.model_dir):
            print(f"❌ Dossier {self.model_dir} introuvable")
            return []
        
        models = []
        metadata_files = [f for f in os.listdir(self.model_dir) if f.startswith("model_metadata_")]
        
        for metadata_file in metadata_files:
            try:
                with open(os.path.join(self.model_dir, metadata_file), 'rb') as f:
                    metadata = pickle.load(f)
                    models.append({
                        'timestamp': metadata['timestamp'],
                        'type': metadata['model_type'],
                        'score': metadata['test_score'],
                        'classes': metadata['num_classes'],
                        'device': metadata['device'],
                        'metadata_file': metadata_file
                    })
            except Exception as e:
                print(f"⚠️  Erreur lecture {metadata_file}: {e}")
        
        return sorted(models, key=lambda x: x['timestamp'], reverse=True)
    
    def load_latest_model(self):
        """Charge le modèle le plus récent"""
        models = self.list_available_models()
        if not models:
            print("❌ Aucun modèle trouvé")
            return False
        
        latest = models[0]
        print(f"📚 Chargement du modèle le plus récent: {latest['timestamp']}")
        return self.load_model_by_timestamp(latest['timestamp'])
    
    def load_model_by_timestamp(self, timestamp):
        """Charge un modèle par son timestamp"""
        try:
            # Charger les métadonnées
            metadata_file = f"{self.model_dir}/model_metadata_{timestamp}.pkl"
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Charger le label encoder
            encoder_file = f"{self.model_dir}/label_encoder_{timestamp}.pkl"
            with open(encoder_file, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            model_type = self.metadata['model_type']
            
            if model_type == 'pytorch_gpu' and TORCH_AVAILABLE:
                return self._load_pytorch_model(timestamp)
            elif model_type in ['ensemble_cpu', 'randomforest_cpu']:
                return self._load_cpu_model(timestamp)
            else:
                print(f"❌ Type de modèle non supporté: {model_type}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur chargement modèle {timestamp}: {e}")
            return False
    
    def _load_pytorch_model(self, timestamp):
        """Charge un modèle PyTorch"""
        try:
            # Charger les configurations TF-IDF
            tfidf_file = f"{self.model_dir}/tfidf_configs_{timestamp}.pkl"
            with open(tfidf_file, 'rb') as f:
                self.tfidf_configs = pickle.load(f)
            
            # Charger le modèle PyTorch avec gestion des versions
            model_file = f"{self.model_dir}/pytorch_model_{timestamp}.pth"
            
            # Essayer d'abord avec weights_only=False (compatible avec anciennes versions)
            try:
                checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            except TypeError:
                # Fallback pour les versions plus anciennes de PyTorch
                checkpoint = torch.load(model_file, map_location='cpu')
            except Exception as e:
                print(f"⚠️  Tentative de chargement alternatif: {e}")
                # Tentative avec allowlist des globals pour numpy
                try:
                    # Pour PyTorch 2.6+ avec weights_only=True
                    if hasattr(torch.serialization, 'add_safe_globals'):
                        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                    checkpoint = torch.load(model_file, map_location='cpu', weights_only=True)
                except:
                    # Dernière tentative: forcer le chargement legacy
                    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            
            config = checkpoint['model_config']
            self.model = TextClassifierNet(
                config['input_size'],
                config['hidden_size'],
                config['num_classes'],
                config['dropout_rate']
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"✅ Modèle PyTorch chargé (score: {checkpoint['test_score']:.3f})")
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement PyTorch: {e}")
            print(f"💡 Suggestion: Regénérez le modèle avec la version actuelle de PyTorch")
            return False
    
    def _load_cpu_model(self, timestamp):
        """Charge un modèle CPU (ensemble ou RandomForest)"""
        try:
            model_type = self.metadata['model_type']
            if model_type == 'ensemble_cpu':
                model_file = f"{self.model_dir}/ensemble_model_{timestamp}.pkl"
            else:
                model_file = f"{self.model_dir}/randomforest_model_{timestamp}.pkl"
            
            self.pipeline = joblib.load(model_file)
            
            score = self.metadata['test_score']
            print(f"✅ Modèle CPU {model_type} chargé (score: {score:.3f})")
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement CPU: {e}")
            return False
    
    def predict(self, texts):
        """Fait des prédictions sur une liste de textes"""
        if not self.is_loaded():
            print("❌ Aucun modèle chargé")
            return None
        
        # Nettoyer les textes
        if isinstance(texts, str):
            texts = [texts]
        
        cleaned_texts = [clean_text(text) for text in texts]
        
        try:
            if self.metadata['model_type'] == 'pytorch_gpu':
                return self._predict_pytorch(cleaned_texts)
            else:
                return self._predict_cpu(cleaned_texts)
        except Exception as e:
            print(f"❌ Erreur prédiction: {e}")
            return None
    
    def _predict_pytorch(self, texts):
        """Prédiction avec modèle PyTorch"""
        # Préparer les features TF-IDF
        X_features = []
        for tfidf in self.tfidf_configs:
            X_tfidf = tfidf.transform(texts).toarray()
            X_features.append(X_tfidf)
        
        X_combined = np.hstack(X_features)
        X_tensor = torch.FloatTensor(X_combined)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_tensor)
            _, predicted = torch.max(output.data, 1)
            predictions = predicted.numpy()
        
        # Convertir en labels
        return self.label_encoder.inverse_transform(predictions)
    
    def _predict_cpu(self, texts):
        """Prédiction avec modèle CPU"""
        predictions = self.pipeline.predict(texts)
        return self.label_encoder.inverse_transform(predictions)
    
    def is_loaded(self):
        """Vérifie si un modèle est chargé"""
        return (self.model is not None or self.pipeline is not None) and self.label_encoder is not None
    
    def get_model_info(self):
        """Retourne les informations du modèle chargé"""
        if not self.is_loaded():
            return None
        
        return {
            'type': self.metadata['model_type'],
            'timestamp': self.metadata['timestamp'],
            'score': self.metadata['test_score'],
            'classes': self.metadata['num_classes'],
            'device': self.metadata['device'],
            'training_samples': self.metadata['training_samples']
        }

def demo_usage():
    """Démonstration d'utilisation"""
    print("🔍 Démonstration du chargeur de modèles")
    print("=" * 50)
    
    loader = ModelLoader()
    
    # Lister les modèles disponibles
    print("📚 Modèles disponibles:")
    models = loader.list_available_models()
    
    if not models:
        print("❌ Aucun modèle trouvé. Exécutez d'abord main.py pour créer un modèle.")
        return
    
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model['timestamp']} - {model['type']} (score: {model['score']:.3f})")
    
    # Charger le modèle le plus récent
    if loader.load_latest_model():
        info = loader.get_model_info()
        print(f"\n📊 Modèle chargé: {info['type']} du {info['timestamp']}")
        print(f"   Score: {info['score']:.3f}, Classes: {info['classes']}, Device: {info['device']}")
        
        # Exemples de prédictions
        exemples = [
            "Table en bois de chêne 120x80 cm",
            "Chaise ergonomique noire en cuir",
            "Lampe de bureau LED blanche",
            "Canapé 3 places gris anthracite",
            "Étagère murale 5 niveaux"
        ]
        
        print(f"\n🎯 Test de prédictions:")
        predictions = loader.predict(exemples)
        
        for text, pred in zip(exemples, predictions):
            print(f"   '{text}' → {pred}")
        
        print(f"\n✅ Démonstration terminée!")
    
if __name__ == "__main__":
    demo_usage()
