import re
from color_mapping import color_mapping, color_adjectives, adjective_normalization

def clean_text(text):
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

def extract_colors(text):
    text = str(text).lower()  # Convert to string first, then to lowercase
    
    found_colors = set()  # Utilise un set pour éviter les doublons
    
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
