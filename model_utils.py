"""
Module pour la définition et les utilitaires du modèle PyTorch.
Contient la classe TextClassifierNet et les fonctions d'affichage de progression.
"""

import torch
import torch.nn as nn

class TextClassifierNet(nn.Module):
    """
    Réseau de neurones avancé pour la classification de texte - Version exacte du backup.
    Architecture sophistiquée avec connexions résiduelles et diverses fonctions d'activation.
    """
    
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.4):
        """
        Initialise le réseau de neurones avancé.
        
        Args:
            input_size (int): Taille d'entrée (nombre de features TF-IDF)
            hidden_size (int): Taille de la couche cachée
            num_classes (int): Nombre de classes de sortie
            dropout_rate (float): Taux de dropout pour la régularisation
        """
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
        """
        Passe avant du réseau avec connexions résiduelles et activations variées.
        
        Args:
            x (torch.Tensor): Tensor d'entrée
            
        Returns:
            torch.Tensor: Sortie du réseau (logits)
        """
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
