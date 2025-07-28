"""
Module pour la définition et les utilitaires du modèle PyTorch.
Contient la classe TextClassifierNet et les fonctions d'affichage de progression.
"""

import torch
import torch.nn as nn

class TextClassifierNet(nn.Module):
    """
    Réseau de neurones pour la classification de texte.
    Architecture : Input -> FC1 -> BatchNorm -> ReLU -> Dropout -> 
                  FC2 -> BatchNorm -> ReLU -> Dropout -> 
                  FC3 -> BatchNorm -> ReLU -> Dropout -> 
                  Output
    """
    
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.4):
        """
        Initialise le réseau de neurones.
        
        Args:
            input_size (int): Taille d'entrée (nombre de features TF-IDF)
            hidden_size (int): Taille de la couche cachée
            num_classes (int): Nombre de classes de sortie
            dropout_rate (float): Taux de dropout pour la régularisation
        """
        super(TextClassifierNet, self).__init__()
        
        # Architecture en couches avec batch normalization et dropout
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Passe avant du réseau.
        
        Args:
            x (torch.Tensor): Tensor d'entrée
            
        Returns:
            torch.Tensor: Sortie du réseau (logits)
        """
        # Première couche
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Deuxième couche
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Troisième couche
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        # Couche de sortie
        x = self.fc4(x)
        
        return x

def print_progress(step, description):
    """
    Affiche une barre de progression formatée.
    
    Args:
        step (int): Numéro de l'étape actuelle
        description (str): Description de l'étape
    """
    print(f"\n{'='*50}")
    print(f"📋 ÉTAPE {step}: {description.upper()}")
    print(f"{'='*50}")

def print_configuration(config):
    """
    Affiche la configuration du modèle de façon formatée.
    
    Args:
        config (dict): Dictionnaire de configuration
    """
    print("🔧 Configuration du modèle:")
    for key, value in config.items():
        print(f"   {key}: {value}")
