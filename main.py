import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import time
import os
from datetime import datetime
from numpy import float32
import numpy as np
import pickle
import joblib

# === CONFIGURATION GPU/CPU ===
USE_GPU = True  # Mettre True pour essayer d'utiliser le GPU (nécessite des dépendances supplémentaires)
USE_ENSEMBLE = True  # Utiliser plusieurs modèles pour améliorer la précision

if USE_GPU:
    try:
        print("🔥 Tentative d'importation des bibliothèques GPU")
        # Tentative d'importation des bibliothèques GPU
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.linear_model import SGDClassifier
        print("🔥 Mode GPU activé - Utilisation de PyTorch")
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎯 Device détecté: {DEVICE}")
        
        # Classe de réseau de neurones avancé pour GPU
        class TextClassifierNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.4):
                super(TextClassifierNet, self).__init__()
                # Architecture plus profonde et sophistiquée
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
                self.fc5 = nn.Linear(hidden_size // 4, num_classes)
                
                # Normalisation et régularisation
                self.dropout = nn.Dropout(dropout_rate)
                self.batch_norm1 = nn.BatchNorm1d(hidden_size)
                self.batch_norm2 = nn.BatchNorm1d(hidden_size)
                self.batch_norm3 = nn.BatchNorm1d(hidden_size // 2)
                self.batch_norm4 = nn.BatchNorm1d(hidden_size // 4)
                
                # Fonctions d'activation variées
                self.relu = nn.ReLU()
                self.leaky_relu = nn.LeakyReLU(0.1)
                self.elu = nn.ELU()
                
                # Initialisation des poids
                self._initialize_weights()
                
            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                
            def forward(self, x):
                # Couche 1
                x = self.fc1(x)
                x = self.batch_norm1(x)
                x = self.relu(x)
                x = self.dropout(x)
                
                # Couche 2 avec connexion résiduelle
                residual = x
                x = self.fc2(x)
                x = self.batch_norm2(x)
                x = self.leaky_relu(x)
                x = self.dropout(x)
                x = x + residual  # Connexion résiduelle
                
                # Couche 3
                x = self.fc3(x)
                x = self.batch_norm3(x)
                x = self.elu(x)
                x = self.dropout(x)
                
                # Couche 4
                x = self.fc4(x)
                x = self.batch_norm4(x)
                x = self.relu(x)
                x = self.dropout(x)
                
                # Couche de sortie
                x = self.fc5(x)
                return x
                
    except ImportError:
        print("⚠️  Bibliothèques GPU non trouvées - Utilisation du CPU")
        print("💡 Pour installer GPU: pip install torch torchvision")
        USE_GPU = False

# === CONFIGURATION DE PERFORMANCE ===
# Utilisation de tous les cœurs CPU disponibles
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

# Configuration pour joblib (parallélisation scikit-learn)
from joblib import parallel_backend

def print_progress(step, message):
    """Affiche le progrès avec un indicateur visuel"""
    print(f"[{step}/8] {message}...")

print("🚀 Démarrage de l'analyse e-commerce")
print("=" * 50)

# === INFORMATION GPU ===
if not USE_GPU:
    print("💡 Pour activer l'accélération GPU:")
    print("   1. Installer PyTorch: pip install torch torchvision")
    print("   2. Changer USE_GPU = True dans le script")
    print("   🔥 Le GPU utilisera un réseau de neurones PyTorch (meilleure précision)")
    print("   � CPU utilise RandomForest (très bonne précision)")
    print("-" * 50)

# === 1. CHARGEMENT DES DONNÉES ===
print_progress(1, "Chargement des données")

# Option pour limiter le nombre de lignes (utile pour les tests)
LIMIT_ROWS = None  # Limiter pour éviter les problèmes de mémoire GPU

xlsb_path = "ecommerce_corrected_20250728_174305.xlsx"
NATURE_COL = "Nature"  # Colonne des catégories
LIBELLE_COL = "Libellé produit"  # Colonne des libellés de produits
SHEET_NAME = "Sheet1"  # Nom de la feuille par défaut
# SHEET_NAME = "20210614 Ecommerce sales"
if LIMIT_ROWS:
    # Première lecture pour connaître toutes les catégories
    df_sample = pd.read_excel(xlsb_path, sheet_name=SHEET_NAME, engine="calamine", nrows=1000)
    df_sample = df_sample[[LIBELLE_COL, NATURE_COL]].dropna()
    all_categories = df_sample[NATURE_COL].unique()

    # Chargement limité
    df = pd.read_excel(xlsb_path, sheet_name=SHEET_NAME, engine="calamine", nrows=LIMIT_ROWS)
    df = df[[LIBELLE_COL, NATURE_COL]].dropna()

    # Vérification et ajout des catégories manquantes
    current_categories = df[NATURE_COL].unique()
    missing_categories = set(all_categories) - set(current_categories)
    
    if missing_categories:
        print(f"⚠️  Catégories manquantes détectées: {len(missing_categories)}")
        # Rechargement du fichier pour récupérer les catégories manquantes
        df_full = pd.read_excel(xlsb_path, sheet_name=SHEET_NAME, engine="calamine")
        df_full = df_full[[LIBELLE_COL, NATURE_COL]].dropna()

        # Ajout d'exemples de chaque catégorie manquante
        for category in missing_categories:
            category_samples = df_full[df_full[NATURE_COL] == category].head(5)  # 5 exemples par catégorie
            df = pd.concat([df, category_samples], ignore_index=True)
        
        print(f"✅ Ajout de {len(missing_categories)} catégories manquantes")
    
    print(f"⚠️  Mode test: chargement limité à {LIMIT_ROWS} lignes + toutes les catégories")
else:
    df = pd.read_excel(xlsb_path, sheet_name=SHEET_NAME, engine="calamine")
    df = df[[LIBELLE_COL, NATURE_COL]].dropna()
    print("📊 Chargement complet du fichier")

print(f"✅ Données chargées: {len(df)} produits trouvés avec {len(df[NATURE_COL].unique())} catégories uniques")

# Reset des indices après toutes les opérations de chargement
df = df.reset_index(drop=True)

# === 2. NETTOYAGE DES TEXTES SIMPLE ET EFFICACE ===
print_progress(2, "Nettoyage des textes")

def clean_text(text):
    """Nettoyage amélioré qui préserve les dimensions cruciales pour la classification"""
    text = str(text).lower()
    
    # Préservation des patterns importants AVANT nettoyage
    # Remplacement des caractères spéciaux par des mots-clés
    text = re.sub(r'&', ' et ', text)
    text = re.sub(r'%', ' pourcent ', text)
    text = re.sub(r'\+', ' plus ', text)
    text = re.sub(r'@', ' arobase ', text)
    
    # PRÉSERVATION INTELLIGENTE DES DIMENSIONS (traitement séquentiel pour éviter les conflits)
    # D'abord traiter les 3D, puis marquer les zones traitées, puis traiter les 2D restants
    
    # Étape 1: Marquer temporairement les dimensions 3D
    text = re.sub(r'(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)(?:\s*cm)?', 
                  r' DIM3D_\1x\2x\3 ', text)
    
    # Étape 2: Traiter les dimensions 2D restantes (qui ne font pas partie d'une 3D)
    text = re.sub(r'(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)(?:\s*cm)?', 
                  r' dim_\1x\2 ', text)
    
    # Étape 3: Restaurer les dimensions 3D avec le bon préfixe
    text = re.sub(r'DIM3D_(\d+(?:[,\.]\d+)?x\d+(?:[,\.]\d+)?x\d+(?:[,\.]\d+)?)', 
                  r'dim_\1', text)
    
    # Préservation des tailles de vêtements
    text = re.sub(r'\b(xs|s|m|l|xl|xxl|xxxl)\b', r' taille_\1 ', text)
    
    # Préservation des mesures avec unités (APRÈS traitement des dimensions principales)
    text = re.sub(r'\b(\d+(?:[,\.]\d+)?)\s*(cm|mm|m|kg|g|ml|l)\b', r' mesure_\1_\2 ', text)
    
    # Normalisation des points et virgules dans les dimensions preservées
    text = re.sub(r'dim_(\d+)[,\.](\d+)', r'dim_\1point\2', text)
    
    # Nettoyage des mots orphelins comme "x" restants
    text = re.sub(r'\s+[xX×*]\s+', ' ', text)
    
    # Nettoyage des caractères spéciaux mais préservation des espaces et underscores
    text = re.sub(r'[^\w\s_]', ' ', text)
    
    # Suppression des mots très courts qui n'apportent pas d'information (SAUF les dimensions)
    words = text.split()
    words = [word for word in words if len(word) >= 2 or word.startswith('dim_')]
    
    # Normalisation des espaces
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Application du nettoyage simple
df["clean_libelle"] = df[LIBELLE_COL].apply(clean_text)
print("✅ Textes nettoyés")

# === 3. ENCODAGE DES CATÉGORIES ===
print_progress(3, "Encodage des catégories")
le = LabelEncoder()
df["nature_encoded"] = le.fit_transform(df[NATURE_COL])
print(f"✅ Catégories encodées: {len(le.classes_)} catégories uniques")

# === 4. ENTRAÎNEMENT DU MODÈLE SIMPLE ET RAPIDE ===
print_progress(4, "Division des données et entraînement du modèle")

# Filtrage des catégories rares pour améliorer les performances
min_samples_per_category = 1  # Augmenté pour de meilleures performances
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
    
    # Re-encodage des catégories filtrées
    le_filtered = LabelEncoder()
    df_filtered["nature_encoded"] = le_filtered.fit_transform(df_filtered[NATURE_COL])

    # Division des données avec stratification possible
    X_train, X_test, y_train, y_test = train_test_split(
        df_filtered["clean_libelle"], df_filtered["nature_encoded"], 
        test_size=0.2, random_state=42, stratify=df_filtered["nature_encoded"]
    )
    
else:
    print("📊 Toutes les catégories ont assez d'exemples")
    le_filtered = le
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_libelle"], df["nature_encoded"], test_size=0.2, random_state=42
    )

# Configuration optimisée pour la performance
print(f"🚀 Utilisation de {os.cpu_count()} cœurs CPU pour l'accélération")

if USE_GPU and 'torch' in globals():
    # Pipeline GPU-optimisé avec réseau de neurones PyTorch + Ensemble
    print("⚡ Configuration GPU: Réseau de neurones PyTorch haute précision + Ensemble")
    
    # TF-IDF optimisé pour meilleure précision (réduit pour GPU)
    tfidf_configs = [
        # Configuration 1: Features générales (réduit)
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
            norm='l2'
        ),
        # Configuration 2: Focus sur les caractères (réduit)
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
            norm='l2'
        )
    ]
    
    # Entraînement de plusieurs TF-IDF
    print("📊 Calcul des features TF-IDF multi-configurations...")
    X_train_features = []
    X_test_features = []
    
    for i, tfidf in enumerate(tfidf_configs):
        print(f"   Configuration {i+1}: {tfidf.analyzer} n-grams {tfidf.ngram_range}")
        X_train_tfidf = tfidf.fit_transform(X_train).toarray()
        X_test_tfidf = tfidf.transform(X_test).toarray()
        X_train_features.append(X_train_tfidf)
        X_test_features.append(X_test_tfidf)
    
    # Combinaison des features
    import numpy as np
    X_train_combined = np.hstack(X_train_features)
    X_test_combined = np.hstack(X_test_features)
    print(f"   Features combinées: {X_train_combined.shape[1]} dimensions")
    
    # Configuration du modèle PyTorch optimisé pour GPU
    input_size = X_train_combined.shape[1]
    hidden_size = min(1024, input_size // 4)  # Architecture plus raisonnable pour GPU
    num_classes = len(le_filtered.classes_)
    
    # Utiliser des batches plus petits pour économiser la mémoire GPU
    batch_size = 64  # Réduit pour éviter OOM
    
    # Entraînement d'un seul modèle optimisé au lieu d'ensemble pour économiser la mémoire
    print(f"🔥 Entraînement modèle optimisé sur GPU ({DEVICE}) avec {num_classes} catégories...")
    print(f"📊 Architecture: {input_size} → {hidden_size} → {hidden_size//2} → {hidden_size//4} → {num_classes}")
    print(f"⚙️  Paramètres: Batch={batch_size}, LR=0.001, Dropout=0.4, Label Smoothing=0.1")
    print(f"🎯 Objectif: Maximiser la précision (actuel: validation tous les 5 époques)")
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
        lr=0.001, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Utiliser des batches plus petits pour économiser la mémoire GPU
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_combined).to(DEVICE),
        torch.LongTensor(y_train.values).to(DEVICE)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Préparer les tenseurs de test
    X_test_tensor = torch.FloatTensor(X_test_combined).to(DEVICE)
    y_test_tensor = torch.LongTensor(y_test.values).to(DEVICE)
        
    # Entraînement avancé avec un seul modèle optimisé
    model.train()
    best_accuracy = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(100):  # Plus d'époques pour un seul modèle
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
        if epoch % 5 == 0 or epoch >= 70:  # Validation tous les 5 époques
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
                
                if patience_counter >= 3 and epoch >= 35:  # Early stopping plus patient
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
        def __init__(self, model, tfidf_configs, device):
            self.model = model
            self.tfidf_configs = tfidf_configs
            self.device = device
            
        def predict(self, X):
            # Préparer les features pour chaque configuration TF-IDF
            X_features = []
            for tfidf in self.tfidf_configs:
                X_tfidf = tfidf.transform(X).toarray()
                X_features.append(X_tfidf)
            
            X_combined = np.hstack(X_features)
            X_tensor = torch.FloatTensor(X_combined).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                output = self.model(X_tensor)
                _, predicted = torch.max(output.data, 1)
                return predicted.cpu().numpy()
    
    # Créer le wrapper
    pipeline = SingleModelWrapper(model, tfidf_configs, DEVICE)
    
    # Score final
    final_predictions = pipeline.predict(X_test)
    test_score = (final_predictions == y_test.values).mean()
    
else:
    # Pipeline CPU-optimisé avec ensemble (RandomForest + XGBoost + SVM)
    print("🔧 Configuration CPU: Ensemble RandomForest + Modèles additionnels")
    
    if USE_ENSEMBLE:
        # Configuration TF-IDF optimisée pour ensemble
        tfidf_ensemble = TfidfVectorizer(
            max_features=7000,  # Plus de features pour l'ensemble
            ngram_range=(1, 3),  # Trigrams pour plus de contexte
            min_df=2,
            max_df=0.9,
            dtype=float32,
            sublinear_tf=True,
            analyzer='word',
            token_pattern=r'\b[a-zA-Z]{2,}\b',
            stop_words=None,
            use_idf=True,
            smooth_idf=True,
            norm='l2'
        )
        
        # Calcul des features
        print("📊 Calcul des features TF-IDF pour ensemble...")
        X_train_tfidf = tfidf_ensemble.fit_transform(X_train)
        X_test_tfidf = tfidf_ensemble.transform(X_test)
        
        from sklearn.svm import SVC
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
        
        # Création de plusieurs modèles
        models_ensemble = [
            ("rf", RandomForestClassifier(
                n_estimators=800,  # Plus d'arbres
                random_state=42,
                n_jobs=-1,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                class_weight='balanced',
                criterion='gini'
            )),
            ("et", ExtraTreesClassifier(
                n_estimators=600,
                random_state=42,
                n_jobs=-1,
                max_depth=None,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                class_weight='balanced',
                criterion='gini'
            )),
            ("lr", LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
                C=1.0,
                class_weight='balanced',
                solver='saga',
                penalty='elasticnet',
                l1_ratio=0.5
            )),
            ("nb", MultinomialNB(
                alpha=0.1,
                fit_prior=True,
                class_prior=None
            ))
        ]
        
        print(f"🚀 Entraînement ensemble de {len(models_ensemble)} modèles CPU...")
        
        # Voting classifier avec soft voting
        ensemble_classifier = VotingClassifier(
            estimators=models_ensemble,
            voting='soft',  # Utilise les probabilités
            n_jobs=-1
        )
        
        # Pipeline complet
        pipeline = Pipeline([
            ("tfidf", tfidf_ensemble),
            ("ensemble", ensemble_classifier)
        ])
        
        # Entraînement avec parallélisation
        with parallel_backend('threading', n_jobs=-1):
            pipeline.fit(X_train, y_train)
            
        test_score = pipeline.score(X_test, y_test)
        
        # Affichage des scores individuels
        print("📊 Scores individuels des modèles:")
        X_train_tfidf_full = tfidf_ensemble.fit_transform(X_train)
        X_test_tfidf_full = tfidf_ensemble.transform(X_test)
        
        for name, model in models_ensemble:
            model.fit(X_train_tfidf_full, y_train)
            individual_score = model.score(X_test_tfidf_full, y_test)
            print(f"   {name}: {individual_score:.3f}")
        
    else:
        # Configuration standard si ensemble désactivé
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                dtype=float32,
                sublinear_tf=True,
                analyzer='word',
                token_pattern=r'\b[a-zA-Z]{2,}\b',
                stop_words=None,
                use_idf=True,
                smooth_idf=True,
                norm='l2'
            )),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                class_weight='balanced',
                criterion='gini'
            ))
        ])
        
        # Entraînement CPU classique
        with parallel_backend('threading', n_jobs=-1):
            pipeline.fit(X_train, y_train)
        
        test_score = pipeline.score(X_test, y_test)

# Entraînement adapté selon la configuration
start_time = time.time()

# Le modèle est déjà entraîné dans la section précédente
training_time = time.time() - start_time

# Scores de validation adaptés selon le type de modèle
if USE_GPU and 'torch' in globals():
    # PyTorch GPU optimisé
    print(f"✅ Modèle GPU (PyTorch optimisé) entraîné en {training_time:.2f}s")
    print(f"📊 Score test: {test_score:.3f}")
    print(f"📊 Modèle: Réseau de neurones PyTorch avancé (GPU)")
    print(f" Train: {len(X_train)}, Test: {len(X_test)}")
else:
    # CPU Ensemble ou standard
    if USE_ENSEMBLE:
        print(f"✅ Ensemble CPU entraîné en {training_time:.2f}s")
        print(f"📊 Score test ensemble: {test_score:.3f}")
        print(f"📊 Modèle: Ensemble de {len(models_ensemble)} modèles (CPU)")
        print(f" Train: {len(X_train)}, Test: {len(X_test)}")
    else:
        oob_score = pipeline.named_steps['clf'].oob_score_
        print(f"✅ Modèle CPU (RandomForest) entraîné en {training_time:.2f}s")
        print(f"📊 Score test: {test_score:.3f}")
        print(f"📊 Score OOB: {oob_score:.3f}")
        print(f" Train: {len(X_train)}, Test: {len(X_test)}")

# === 5. RECATÉGORISATION DE TOUT LE DATASET ===
print_progress(5, "Prédiction sur l'ensemble du dataset")

# Prédiction avec gestion des catégories rares
start_time = time.time()
batch_size = 10000  # Taille de lot pour la prédiction
total_rows = len(df)
predictions = []

print(f"📦 Traitement par lots de {batch_size} éléments...")

# Prédiction simple et efficace par lots
with parallel_backend('threading', n_jobs=-1):
    for i in range(0, total_rows, batch_size):
        batch_end = min(i + batch_size, total_rows)
        batch_data = df["clean_libelle"].iloc[i:batch_end]
        
        # Prédiction directe avec le pipeline
        batch_predictions = pipeline.predict(batch_data)
        predictions.extend(batch_predictions)
        
        # Affichage du progrès
        progress = (batch_end / total_rows) * 100
        print(f"   Lot {i//batch_size + 1}: {progress:.1f}% terminé ({batch_end}/{total_rows})")

# Conversion des prédictions avec gestion des catégories filtrées
df["predicted_nature_encoded"] = predictions

# Création d'une colonne pour les prédictions avec gestion des catégories rares
predicted_labels = []
for i, pred in enumerate(predictions):
    original_category = df.iloc[i][NATURE_COL]
    
    # Si la catégorie originale n'était pas dans l'entraînement, la marquer comme rare
    if original_category not in valid_categories:
        predicted_labels.append("CATÉGORIE_RARE")
    # Sinon, utiliser la prédiction du modèle
    elif pred < len(le_filtered.classes_):
        predicted_labels.append(le_filtered.inverse_transform([pred])[0])
    else:
        predicted_labels.append("CATÉGORIE_RARE")  # Sécurité pour les prédictions hors limites


PREDICTED_NATURE_COL = "predicted_nature_col"
df[PREDICTED_NATURE_COL] = predicted_labels
prediction_time = time.time() - start_time

# Calcul de la précision en excluant les catégories rares des métriques
valid_mask = df[NATURE_COL].isin(valid_categories)
df_valid = df[valid_mask]

# Ajout d'une colonne VRAI/FAUX pour indiquer si la prédiction est correcte
df["prediction_correcte"] = df.apply(lambda row: 
    "VRAI" if row[NATURE_COL] == row[PREDICTED_NATURE_COL] else "FAUX", axis=1)

if len(df_valid) > 0:
    # Calcul de la précision SEULEMENT sur les catégories que le modèle a apprises
    df_trainable = df[df[NATURE_COL].isin(valid_categories)]
    misclassified_trainable = (df_trainable[NATURE_COL] != df_trainable[PREDICTED_NATURE_COL]).sum()
    accuracy_trainable = ((len(df_trainable) - misclassified_trainable) / len(df_trainable)) * 100
    
    print(f"✅ Prédictions terminées en {prediction_time:.2f}s")
    print(f"📊 Précision (catégories entraînées): {accuracy_trainable:.1f}% ({misclassified_trainable}/{len(df_trainable)} erreurs)")
    
    # Statistiques globales (incluant les catégories rares)
    correct_predictions = (df["prediction_correcte"] == "VRAI").sum()
    total_predictions = len(df)
    global_accuracy = (correct_predictions / total_predictions) * 100
    print(f"📈 Précision globale (toutes catégories): {global_accuracy:.1f}% ({correct_predictions}/{total_predictions} correctes)")
    
    # Statistiques détaillées des catégories rares
    rare_items = df[~df[NATURE_COL].isin(valid_categories)]
    rare_count = len(rare_items)
    if rare_count > 0:
        print(f"⚠️  {rare_count} produits de catégories rares marqués comme 'CATÉGORIE_RARE'")
        print(f"📊 Répartition: {len(df_trainable)} entraînables, {rare_count} rares")
else:
    print("✅ Prédictions terminées - Aucune catégorie valide trouvée")

# === 6. EXTRACTION DES DIMENSIONS & COULEURS ===
print_progress(6, "Extraction des dimensions et couleurs")

# Mapping des variantes de couleurs vers les couleurs de base (source unique)
color_mapping = {
    # Couleurs de base
    "blanc": "blanc", "blanche": "blanc", "blancs": "blanc", "blanches": "blanc",
    "noir": "noir", "noire": "noir", "noirs": "noir", "noires": "noir",
    "gris": "gris", "grise": "gris", "grises": "gris",
    "rouge": "rouge", "rouges": "rouge",
    "bleu": "bleu", "bleue": "bleu", "bleus": "bleu", "bleues": "bleu",
    "vert": "vert", "verte": "vert", "verts": "vert", "vertes": "vert",
    "jaune": "jaune", "jaunes": "jaune",
    "rose": "rose", "roses": "rose",
    "violet": "violet", "violette": "violet", "violets": "violet", "violettes": "violet",
    "orange": "orange", "oranges": "orange",
    "turquoise": "turquoise", "turquoises": "turquoise",
    "cyan": "cyan", "cyans": "cyan",
    "magenta": "magenta", "magentas": "magenta",
    
    # Nuances de brun et bois
    "brun": "brun", "brune": "brun", "bruns": "brun", "brunes": "brun",
    "marron": "marron", "marrons": "marron",
    "bois": "bois", "boisé": "bois", "boisée": "bois", "boisés": "bois", "boisées": "bois",
    "chêne": "chêne", "chênes": "chêne",
    "acajou": "acajou", "acajous": "acajou",
    "teck": "teck", "tecks": "teck",
    "bambou": "bambou", "bambous": "bambou",
    "pin": "pin", "pins": "pin",
    "érable": "érable", "érables": "érable",
    "noyer": "noyer", "noyers": "noyer",
    "hêtre": "hêtre", "hêtres": "hêtre",
    "frêne": "frêne", "frênes": "frêne",
    "merisier": "merisier", "merisiers": "merisier",
    "châtaignier": "châtaignier", "châtaigniers": "châtaignier",
    "cerisier": "cerisier", "cerisiers": "cerisier",
    
    # Couleurs pastel et nuances
    "beige": "beige", "beiges": "beige",
    "crème": "crème", "crèmes": "crème",
    "ivoire": "ivoire", "ivoires": "ivoire",
    "écru": "écru", "écrus": "écru",
    "taupe": "taupe", "taupes": "taupe",
    "sable": "sable", "sables": "sable",
    "ocre": "ocre", "ocres": "ocre",
    "terre": "terre", "terres": "terre",
    "camel": "camel", "camels": "camel",
    "café": "café", "cafés": "café",
    "chocolat": "chocolat", "chocolats": "chocolat",
    "cognac": "cognac", "cognacs": "cognac",
    "caramel": "caramel", "caramels": "caramel",
    "miel": "miel", "miels": "miel",
    
    # Couleurs métalliques
    "argent": "argent", "argenté": "argent", "argentée": "argent", "argentés": "argent", "argentées": "argent",
    "or": "or", "doré": "or", "dorée": "or", "dorés": "or", "dorées": "or",
    "bronze": "bronze", "bronzé": "bronze", "bronzée": "bronze", "bronzés": "bronze", "bronzées": "bronze",
    "cuivre": "cuivre", "cuivré": "cuivre", "cuivrée": "cuivre", "cuivrés": "cuivre", "cuivrées": "cuivre",
    "laiton": "laiton", "laitons": "laiton",
    "chrome": "chrome", "chromé": "chrome", "chromée": "chrome", "chromés": "chrome", "chromées": "chrome",
    "nickel": "nickel", "nickels": "nickel",
    "platine": "platine", "platines": "platine",
    
    # Couleurs spéciales
    "transparent": "transparent", "transparente": "transparent", "transparents": "transparent", "transparentes": "transparent",
    "opaque": "opaque", "opaques": "opaque",
    "mat": "mat", "mate": "mat", "mats": "mat", "mates": "mat",
    "brillant": "brillant", "brillante": "brillant", "brillants": "brillant", "brillantes": "brillant",
    "satiné": "satiné", "satinée": "satiné", "satinés": "satiné", "satinées": "satiné",
    "anthracite": "anthracite", "anthracites": "anthracite",
    "charbon": "charbon", "charbons": "charbon",
    "ardoise": "ardoise", "ardoises": "ardoise",
    "graphite": "graphite", "graphites": "graphite",
    
    # Couleurs nature
    "naturel": "naturel", "naturelle": "naturel", "naturels": "naturel", "naturelles": "naturel",
    "brut": "brut", "brute": "brut", "bruts": "brut", "brutes": "brut",
    "rustique": "rustique", "rustiques": "rustique",
    "vintage": "vintage", "vintages": "vintage",
    "antique": "antique", "antiques": "antique",
    "vieilli": "vieilli", "vieillie": "vieilli", "vieillis": "vieilli", "vieillies": "vieilli",
    
    # Couleurs vives
    "fluo": "fluo", "fluos": "fluo",
    "néon": "néon", "néons": "néon",
    "phosphorescent": "phosphorescent", "phosphorescente": "phosphorescent", "phosphorescents": "phosphorescent", "phosphorescentes": "phosphorescent"
}

def extract_dimensions(text):
    text = str(text)
    
    # Patterns pour différents formats de dimensions
    patterns = [
        # Format 3D avec cm: "108 x 32 5 x 48 cm" -> "108 x 32.5 x 48"
        r'(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s*cm',
        # Format 3D standard: "143 x 36 x 178 cm" ou "143 x 36 x 178"
        r'(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)',
        # Format 2D: "45 x 75"
        r'(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            dimensions = []
            for match in matches:
                if len(match) == 4:  # Format spécial "108 x 32 5 x 48"
                    # Combine le 2e et 3e groupe: "32" + "5" = "32.5"
                    dim1 = match[0].replace(',', '.')
                    dim2 = match[1] + '.' + match[2]  # "32" + "." + "5" = "32.5"
                    dim3 = match[3].replace(',', '.')
                    dimensions.append(f"{dim1}*{dim2}*{dim3}")
                elif len(match) == 3:  # Format 3D standard
                    dim_parts = [part.replace(',', '.') for part in match if part]
                    if len(dim_parts) == 3:
                        dimensions.append("*".join(dim_parts))
                elif len(match) == 2:  # Format 2D
                    dim_parts = [part.replace(',', '.') for part in match if part]
                    if len(dim_parts) == 2:
                        dimensions.append("*".join(dim_parts))
            
            if dimensions:
                return " | ".join(dimensions)
    
    return None

def extract_colors(text):
    text = str(text).lower()  # Convert to string first, then to lowercase
    
    found_colors = set()  # Utilise un set pour éviter les doublons
    
    # Liste des adjectifs de couleur courants
    color_adjectives = [
        "clair", "claire", "clairs", "claires",
        "foncé", "foncée", "foncés", "foncées", 
        "sombre", "sombres",
        "pale", "pâle", "pales", "pâles",
        "vif", "vive", "vifs", "vives",
        "intense", "intenses",
        "pastel", "pastels",
        "tendre", "tendres",
        "doux", "douce", "douces",
        "léger", "légère", "légers", "légères",
        "profond", "profonde", "profonds", "profondes"
    ]
    
    # Dictionnaire pour normaliser les adjectifs vers leur forme de base
    adjective_normalization = {
        "claire": "clair", "clairs": "clair", "claires": "clair",
        "foncée": "foncé", "foncés": "foncé", "foncées": "foncé",
        "pâle": "pâle", "pales": "pâle", "pâles": "pâle",
        "vive": "vif", "vifs": "vif", "vives": "vif",
        "douce": "doux", "douces": "doux",
        "légère": "léger", "légers": "léger", "légères": "léger",
        "profonde": "profond", "profonds": "profond", "profondes": "profond"
    }
    
    # Préparation des patterns
    all_color_variants = '|'.join(re.escape(couleur) for couleur in color_mapping.keys())
    adj_variants = '|'.join(re.escape(adj) for adj in color_adjectives)
    
    # 1. D'ABORD: chercher les couleurs avec adjectifs (priorité haute)
    combined_pattern = rf'\b(?:({all_color_variants})\s+({adj_variants})|({adj_variants})\s+({all_color_variants}))\b'
    
    # Collecter toutes les positions des matches avec adjectifs
    adjective_matches = []
    for match in re.finditer(combined_pattern, text):
        if match.group(1) and match.group(2):  # couleur + adjectif (ex: "gris clair")
            couleur_trouvee = match.group(1)
            adjectif_trouve = match.group(2)
            base_color = color_mapping[couleur_trouvee]
            # Normaliser l'adjectif avec le dictionnaire
            adj_normalized = adjective_normalization.get(adjectif_trouve, adjectif_trouve)
            
            color_with_adj = f"{base_color} {adj_normalized}"
            found_colors.add(color_with_adj)
            # Enregistrer la position pour éviter les doublons
            adjective_matches.append((match.start(), match.end()))
            
        elif match.group(3) and match.group(4):  # adjectif + couleur (ex: "clair gris")
            couleur_trouvee = match.group(4)
            adjectif_trouve = match.group(3)
            base_color = color_mapping[couleur_trouvee]
            # Normaliser l'adjectif avec le dictionnaire
            adj_normalized = adjective_normalization.get(adjectif_trouve, adjectif_trouve)
            
            color_with_adj = f"{base_color} {adj_normalized}"
            found_colors.add(color_with_adj)
            # Enregistrer la position pour éviter les doublons
            adjective_matches.append((match.start(), match.end()))
    
    # 2. ENSUITE: chercher les couleurs simples (en évitant les zones déjà trouvées)
    simple_pattern = rf'\b({all_color_variants})\b'
    
    for match in re.finditer(simple_pattern, text):
        # Vérifier si cette couleur n'est pas déjà dans une combinaison avec adjectif
        is_in_adjective_match = False
        for adj_start, adj_end in adjective_matches:
            if adj_start <= match.start() < adj_end or adj_start < match.end() <= adj_end:
                is_in_adjective_match = True
                break
        
        # Ajouter seulement si ce n'est pas déjà trouvé avec un adjectif
        if not is_in_adjective_match:
            couleur_trouvee = match.group(1)
            base_color = color_mapping[couleur_trouvee]
            found_colors.add(base_color)
    
    return ", ".join(sorted(found_colors))

df["dimension_extraite"] = df[LIBELLE_COL].apply(extract_dimensions)
df["couleur_extraite"] = df[LIBELLE_COL].apply(extract_colors)

# Statistiques d'extraction
dimensions_found = df["dimension_extraite"].notna().sum()
colors_found = (df["couleur_extraite"] != "").sum()
print(f"✅ Extraction terminée - Dimensions: {dimensions_found}, Couleurs: {colors_found}")

# === 7. EXPORT DES RÉSULTATS ===
print_progress(7, "Export des résultats")

# Génération d'un nom de fichier unique avec timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"resultats_ecommerce_analyses_{timestamp}.xlsx"

# Vérification que le fichier n'existe pas déjà (sécurité supplémentaire)
counter = 1
base_filename = output_filename
while os.path.exists(output_filename):
    name_part = base_filename.replace('.xlsx', '')
    output_filename = f"{name_part}_{counter}.xlsx"
    counter += 1

df.to_excel(output_filename, index=False)

print(f"✅ Script terminé. Fichier '{output_filename}' généré.")

# === 8. SAUVEGARDE DU MODÈLE ===
print_progress(8, "Sauvegarde du modèle")

# Création du dossier de sauvegarde
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Génération des noms de fichiers avec timestamp
model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

try:
    if USE_GPU and 'torch' in globals():
        # Sauvegarde modèle PyTorch GPU
        model_filename = f"{model_dir}/pytorch_model_{model_timestamp}.pth"
        tfidf_filename = f"{model_dir}/tfidf_configs_{model_timestamp}.pkl"
        label_encoder_filename = f"{model_dir}/label_encoder_{model_timestamp}.pkl"
        metadata_filename = f"{model_dir}/model_metadata_{model_timestamp}.pkl"
        
        # Sauvegarder le modèle PyTorch
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_classes': num_classes,
                'dropout_rate': 0.4
            },
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
            'timestamp': model_timestamp,
            'test_score': test_score,
            'num_classes': num_classes,
            'valid_categories': valid_categories.tolist(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'use_ensemble': False,
            'device': DEVICE
        }
        with open(metadata_filename, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Modèle PyTorch sauvegardé:")
        print(f"   📁 Modèle: {model_filename}")
        print(f"   📁 TF-IDF: {tfidf_filename}")
        print(f"   📁 Encodeur: {label_encoder_filename}")
        print(f"   📁 Métadonnées: {metadata_filename}")
        
    else:
        # Sauvegarde modèle CPU (ensemble ou standard)
        if USE_ENSEMBLE:
            model_filename = f"{model_dir}/ensemble_model_{model_timestamp}.pkl"
            model_type = 'ensemble_cpu'
        else:
            model_filename = f"{model_dir}/randomforest_model_{model_timestamp}.pkl"
            model_type = 'randomforest_cpu'
        
        label_encoder_filename = f"{model_dir}/label_encoder_{model_timestamp}.pkl"
        metadata_filename = f"{model_dir}/model_metadata_{model_timestamp}.pkl"
        
        # Sauvegarder le pipeline complet (TF-IDF + modèle)
        joblib.dump(pipeline, model_filename)
        
        # Sauvegarder le label encoder
        with open(label_encoder_filename, 'wb') as f:
            pickle.dump(le_filtered, f)
        
        # Sauvegarder les métadonnées
        metadata = {
            'model_type': model_type,
            'timestamp': model_timestamp,
            'test_score': test_score,
            'num_classes': len(le_filtered.classes_),
            'valid_categories': valid_categories.tolist(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'use_ensemble': USE_ENSEMBLE,
            'device': 'cpu'
        }
        if not USE_ENSEMBLE:
            metadata['oob_score'] = pipeline.named_steps['clf'].oob_score_
        
        with open(metadata_filename, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Modèle CPU sauvegardé:")
        print(f"   📁 Pipeline: {model_filename}")
        print(f"   📁 Encodeur: {label_encoder_filename}")
        print(f"   📁 Métadonnées: {metadata_filename}")
    
    # Affichage du résumé de sauvegarde
    print(f"📊 Résumé de la sauvegarde:")
    print(f"   🎯 Type: {metadata['model_type']}")
    print(f"   📈 Score test: {metadata['test_score']:.3f}")
    print(f"   🏷️  Classes: {metadata['num_classes']}")
    print(f"   📚 Échantillons train/test: {metadata['training_samples']}/{metadata['test_samples']}")
    
except Exception as e:
    print(f"❌ Erreur lors de la sauvegarde du modèle: {e}")
    print("⚠️  Le script continue malgré l'erreur de sauvegarde")

print("=" * 50)
print("🎉 Analyse e-commerce terminée avec succès!")
print(f"📁 Modèles sauvegardés dans le dossier: {model_dir}/")
print("💡 Pour réutiliser le modèle, chargez les fichiers correspondants")
