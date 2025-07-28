"""
Module pour le traitement de texte et l'extraction d'informations.
Contient les fonctions de nettoyage de texte, extraction de dimensions et couleurs.
"""

import re

# Dictionnaire de mapping des couleurs
color_mapping = {
    # Couleurs de base
    "rouge": "rouge", "rouges": "rouge",
    "bleu": "bleu", "bleue": "bleu", "bleus": "bleu", "bleues": "bleu",
    "vert": "vert", "verte": "vert", "verts": "vert", "vertes": "vert",
    "jaune": "jaune", "jaunes": "jaune",
    "orange": "orange", "orangé": "orange", "orangée": "orange", "orangés": "orange", "orangées": "orange",
    "violet": "violet", "violette": "violet", "violets": "violet", "violettes": "violet",
    "rose": "rose", "roses": "rose",
    "marron": "marron", "marrons": "marron",
    "brun": "brun", "brune": "brun", "bruns": "brun", "brunes": "brun",
    "beige": "beige", "beiges": "beige",
    "crème": "crème", "crèmes": "crème", "creme": "crème", "cremes": "crème",
    
    # Couleurs neutres
    "blanc": "blanc", "blanche": "blanc", "blancs": "blanc", "blanches": "blanc",
    "noir": "noir", "noire": "noir", "noirs": "noir", "noires": "noir",
    "gris": "gris", "grise": "gris", "grises": "gris",
    "ivoire": "ivoire", "ivoires": "ivoire",
    "écru": "écru", "écrues": "écru", "ecru": "écru", "ecrues": "écru",
    
    # Couleurs de bois
    "chêne": "chêne", "chenes": "chêne", "chene": "chêne", "chenes": "chêne",
    "hêtre": "hêtre", "hetres": "hêtre", "hetre": "hêtre", "hetres": "hêtre",
    "pin": "pin", "pins": "pin",
    "sapin": "sapin", "sapins": "sapin",
    "acajou": "acajou", "acajous": "acajou",
    "noyer": "noyer", "noyers": "noyer",
    "frêne": "frêne", "frenes": "frêne", "frene": "frêne", "frenes": "frêne",
    "érable": "érable", "erables": "érable", "erable": "érable", "erables": "érable",
    "bambou": "bambou", "bambous": "bambou",
    "rotin": "rotin", "rotins": "rotin",
    "teck": "teck", "tecks": "teck",
    "wengé": "wengé", "wenges": "wengé", "wenge": "wengé", "wenges": "wengé",
    "palissandre": "palissandre", "palissandres": "palissandre",
    "eucalyptus": "eucalyptus",
    
    # Couleurs métalliques
    "argent": "argent", "argenté": "argent", "argentée": "argent", "argentés": "argent", "argentées": "argent",
    "or": "or", "doré": "or", "dorée": "or", "dorés": "or", "dorées": "or",
    "bronze": "bronze", "bronzé": "bronze", "bronzée": "bronze", "bronzés": "bronze", "bronzées": "bronze",
    "cuivre": "cuivre", "cuivré": "cuivre", "cuivrée": "cuivre", "cuivrés": "cuivre", "cuivrées": "cuivre",
    "acier": "acier", "aciers": "acier",
    "inox": "inox", "inoxydable": "inox",
    "chrome": "chrome", "chromé": "chrome", "chromée": "chrome", "chromés": "chrome", "chromées": "chrome",
    "laiton": "laiton", "laitons": "laiton",
    "aluminium": "aluminium", "aluminiums": "aluminium",
    "fer": "fer", "fers": "fer",
    "zinc": "zinc", "zincs": "zinc",
    "étain": "étain", "etain": "étain", "etains": "étain",
    
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

def clean_text(text):
    """
    Nettoyage amélioré qui préserve les dimensions cruciales pour la classification.
    Version optimisée identique à l'original.
    
    Args:
        text (str): Texte à nettoyer
        
    Returns:
        str: Texte nettoyé et normalisé
    """
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

def extract_dimensions(text):
    """
    Extrait les dimensions d'un texte.
    
    Args:
        text (str): Texte contenant potentiellement des dimensions
        
    Returns:
        str or None: Dimensions extraites ou None si aucune trouvée
    """
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
    """
    Extrait les couleurs d'un texte.
    
    Args:
        text (str): Texte contenant potentiellement des couleurs
        
    Returns:
        str: Couleurs extraites séparées par des virgules
    """
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
