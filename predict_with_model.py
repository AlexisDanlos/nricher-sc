"""
Script pour utiliser le modèle le plus récent pour prédire la Nature des produits
dans le fichier original 20210614 Ecommerce sales.xlsb
"""

import pandas as pd
import os
import re
from datetime import datetime
from load_model import ModelLoader, clean_text
import numpy as np

# === EXTRACTION DES DIMENSIONS ET COULEURS ===
# Mapping des variantes de couleurs vers les couleurs de base (exactement comme dans main_backup.py)
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
    """
    Extrait les dimensions du texte avec gestion avancée des cas spéciaux.
    Retourne les dimensions sous format standardisé ou None si aucune trouvée.
    """
    text = str(text)
    
    # Exclure les unités de poids et autres unités non-dimensionnelles
    if re.search(r'\d+\s*kg\b', text, re.IGNORECASE):
        # Si le texte contient des kg, filtrer pour ne garder que les vraies dimensions
        text_filtered = re.sub(r'\d+\s*[xX×*]\s*\d+\s*kg\b', '', text, flags=re.IGNORECASE)
        text_filtered = re.sub(r'\d+\s*kg\b', '', text_filtered, flags=re.IGNORECASE)
        text = text_filtered
    
    # Exclure les tailles TV (pouces)
    if re.search(r'tv\s+\d+\s+\d+', text, re.IGNORECASE):
        return None
    
    # Patterns pour différents formats de dimensions (ordre de priorité important)
    patterns = [
        # Format avec espaces décimaux avec 0: "50 0 x 50 0 x 177 8 cm" -> "50.0*50.0*177.8"
        r'(\d+)\s+0\s*[xX×*]\s*(\d+)\s+0\s*[xX×*]\s*(\d+)\s+(\d+)\s*cm',
        # Format dimensions principales 3D: "100x38x38 cm" -> "100*38*38" (priorité haute)
        r'(\d+(?:[,\.]\d+)?)[xX×*](\d+(?:[,\.]\d+)?)[xX×*](\d+(?:[,\.]\d+)?)\s*cm\b',
        # Format dimensions principales 2D: "160x200" -> "160*200" (priorité haute, avant les petits nombres)
        r'\b(\d{2,3})[xX×*](\d{2,3})\b(?!\s*kg)(?!\s*[xX×*]\d+)',
        # Format avec l/h notation avec espaces décimaux: "l 80 x 39 5 x h 75cm" -> "80*39.5*75"
        r'[lh]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*[lh]\s*(\d+(?:[,\.]\d+)?)cm',
        # Format avec l/h notation: "l 43 x l 6 x h 6 cm" -> "43*6*6"
        r'[lh]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*[lh]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*[lh]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format avec diam notation: "diam 35 x h 40 cm" -> "35*40"
        r'diam\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*[lh]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format dimensions avec l/h et espaces décimaux: "80l x 43l x 54 5h cm" -> "80*43*54.5"
        r'(\d+(?:[,\.]\d+)?)[lh]\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)[lh]\s*[xX×*]\s*(\d+)\s+(\d+)[lh]\s*cm',
        # Format dimensions avec l/h sans x: "90l x 42l x 58h cm" -> "90*42*58"
        r'(\d+(?:[,\.]\d+)?)[lh]\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)[lh]\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)[lh]\s*cm',
        # Format avec l/p/h mixtes 3D: "l 120 x p 50 x h 41 cm" -> "120*50*41"
        r'[lph]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*[lph]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*[lph]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format avec l/p mixtes 2D: "l 205 x p 80 cm" -> "205*80"
        r'[lph]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*[lph]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format 3D avec cm: "108 x 32 5 x 48 cm" -> "108 x 32.5 x 48" (HIGH PRIORITY for 4 groups)
        r'(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+)\s+(\d+)\s*[xX×*]\s*(\d+)\s*cm',
        # Format avec espaces décimaux au milieu: "38 5 x 54 cm" -> "38.5*54" (HIGH PRIORITY for 3 groups)
        r'(\d+)\s+(\d+)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format cm après chaque dimension: "112 cm x 207 cm x 57 cm" -> "112*207*57"
        r'(\d+(?:[,\.]\d+)?)\s*cm\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*cm\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format éviter multiplicateurs petits: éviter "2x20" si dimensions principales présentes
        r'(?<!\d[xX×*])(\d{2,})\s+(\d{2,})(?!\s*[xX×*])(?!\s*kg)',
        # Format sommier éviter multiplicateur: "2x90x200" -> "90*200"
        r'\d+[xX×*](\d+(?:[,\.]\d+)?)[xX×*](\d+(?:[,\.]\d+)?)\s*cm',
        # Format avec espaces décimaux sans cm: "3x7 5 m" -> "3*7.5"
        r'(\d+(?:[,\.]\d+)?)[xX×*](\d+)\s+(\d+)\s*m',
        # Format avec espaces décimaux 2D avec m: "1 5 x 10m" -> "1.5*10"
        r'(\d+)\s+(\d+)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)m',
        # Format avec espaces dans les décimaux: "39 5" -> "39.5"
        r'(\d+)\s+(\d)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)',
        # Format unités mixtes: "27 5 cm 3m" -> "27.5*3"
        r'(\d+)\s+(\d)\s*cm\s*(\d+(?:[,\.]\d+)?)m',
        # Format avec cm collé: "200cm x 180 cm" -> "200*180"
        r'(\d+(?:[,\.]\d+)?)cm\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*cm',
        # Format 3D standard: "143 x 36 x 178 cm" ou "143 x 36 x 178"
        r'(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)',
        # Format 2D: "45 x 75" (plus restrictif pour éviter H20 22)
        r'\b(\d+(?:[,\.]\d+)?)\s*[xX×*]\s*(\d+(?:[,\.]\d+)?)\b(?!\s*mousse)(?!\s*H\d+)'
    ]
    
    found_dimensions = []
    main_dimension_found = False
    
    for pattern_idx, pattern in enumerate(patterns):
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            dimensions = []
            for match in matches:
                if len(match) == 4:
                    if pattern_idx == 0:  # Format avec espaces décimaux avec 0
                        dim1 = match[0] + '.0'
                        dim2 = match[1] + '.0'
                        dim3 = match[2] + '.' + match[3]
                        dimensions.append(f"{dim1}*{dim2}*{dim3}")
                        main_dimension_found = True
                    elif pattern_idx == 3:  # Format l/h avec espaces décimaux
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1] + '.' + match[2]  # "39" + "." + "5" = "39.5"
                        dim3 = match[3].replace(',', '.')
                        dimensions.append(f"{dim1}*{dim2}*{dim3}")
                        main_dimension_found = True
                    elif pattern_idx == 6:  # Format l/h avec espaces décimaux à la fin
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1].replace(',', '.')
                        dim3 = match[2] + '.' + match[3]  # "54" + "." + "5" = "54.5"
                        dimensions.append(f"{dim1}*{dim2}*{dim3}")
                        main_dimension_found = True
                    elif pattern_idx == 17:  # Format avec espaces dans les décimaux
                        # "39 5" -> "39.5"
                        dim1 = match[0] + '.' + match[1]
                        dim2 = match[2].replace(',', '.')
                        dim3 = match[3].replace(',', '.')
                        dimensions.append(f"{dim1}*{dim2}*{dim3}")
                    elif pattern_idx == 10:  # Format spécial "108 x 32 5 x 48"
                        # Combine le 2e et 3e groupe: "32" + "5" = "32.5"
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1] + '.' + match[2]  # "32" + "." + "5" = "32.5"
                        dim3 = match[3].replace(',', '.')
                        dimensions.append(f"{dim1}*{dim2}*{dim3}")
                        main_dimension_found = True
                elif len(match) == 3:
                    if pattern_idx in [1, 4, 7, 8, 12, 21]:  # Formats avec l/h/p/cm
                        dim_parts = [part.replace(',', '.') for part in match if part]
                        if len(dim_parts) == 3:
                            # Vérifier si c'est une vraie dimension (pas trop petite)
                            dims = [float(d) for d in dim_parts]
                            if min(dims) >= 10 or pattern_idx == 1:  # Dimensions principales ou format avec cm
                                dimensions.append("*".join(dim_parts))
                                if pattern_idx == 1:  # Format avec cm = priorité haute
                                    main_dimension_found = True
                    elif pattern_idx == 11:  # Format avec espaces décimaux au milieu: "38 5 x 54 cm"
                        dim1 = match[0] + '.' + match[1]  # "38" + "." + "5" = "38.5"
                        dim2 = match[2].replace(',', '.')
                        dimensions.append(f"{dim1}*{dim2}")
                        main_dimension_found = True
                    elif pattern_idx == 15:  # Format avec espaces décimaux sans cm: "3x7 5 m"
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1] + '.' + match[2]  # "7" + "." + "5" = "7.5"
                        dimensions.append(f"{dim1}*{dim2}")
                        main_dimension_found = True
                    elif pattern_idx == 16:  # Format avec espaces décimaux 2D avec m: "1 5 x 10m"
                        dim1 = match[0] + '.' + match[1]  # "1" + "." + "5" = "1.5"
                        dim2 = match[2].replace(',', '.')
                        dimensions.append(f"{dim1}*{dim2}")
                        main_dimension_found = True
                    elif pattern_idx == 18:  # Format unités mixtes "27 5 cm 3m"
                        # "27 5" -> "27.5"
                        dim1 = match[0] + '.' + match[1]
                        dim2 = match[2].replace(',', '.')
                        dimensions.append(f"{dim1}*{dim2}")
                        main_dimension_found = True
                    else:  # Format 3D standard
                        dim_parts = [part.replace(',', '.') for part in match if part]
                        if len(dim_parts) == 3:
                            dimensions.append("*".join(dim_parts))
                elif len(match) == 2:
                    if pattern_idx == 2:  # Format dimensions principales 2D - priorité haute
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1].replace(',', '.')
                        # Vérifier que ce sont des dimensions raisonnables (pas des codes produit)
                        try:
                            d1, d2 = float(dim1), float(dim2)
                            if 10 <= d1 <= 500 and 10 <= d2 <= 500:  # Dimensions réalistes en cm
                                dimensions.append(f"{dim1}*{dim2}")
                                main_dimension_found = True
                        except:
                            pass
                    elif pattern_idx in [5, 9]:  # Format diam ou l/p 2D
                        dim_parts = [part.replace(',', '.') for part in match if part]
                        if len(dim_parts) == 2:
                            dimensions.append("*".join(dim_parts))
                            main_dimension_found = True
                    elif pattern_idx in [13, 14]:  # Format éviter multiplicateurs
                        dim1 = match[0].replace(',', '.')
                        dim2 = match[1].replace(',', '.')
                        # Éviter les petites valeurs qui sont probablement des multiplicateurs
                        try:
                            d1, d2 = float(dim1), float(dim2)
                            if d1 >= 20 and d2 >= 20:  # Seulement les dimensions >= 20
                                dimensions.append(f"{dim1}*{dim2}")
                        except:
                            pass
                    elif pattern_idx == 19:  # Format avec cm collé
                        dim_parts = [part.replace(',', '.') for part in match if part]
                        if len(dim_parts) == 2:
                            dimensions.append("*".join(dim_parts))
                            main_dimension_found = True
                    else:  # Format 2D standard
                        dim_parts = [part.replace(',', '.') for part in match if part]
                        if len(dim_parts) == 2:
                            # Éviter les dimensions trop petites sauf si c'est le seul match
                            try:
                                d1, d2 = float(dim_parts[0]), float(dim_parts[1])
                                if d1 >= 15 and d2 >= 15:  # Seulement dimensions >= 15
                                    dimensions.append("*".join(dim_parts))
                            except:
                                pass
            
            if dimensions:
                found_dimensions.extend(dimensions)
                # Si on a trouvé une dimension principale, arrêter la recherche
                if main_dimension_found and pattern_idx <= 2:
                    break
    
    if found_dimensions:
        # Supprimer les doublons tout en gardant l'ordre
        unique_dimensions = []
        seen = set()
        for dim in found_dimensions:
            if dim not in seen:
                unique_dimensions.append(dim)
                seen.add(dim)
        
        # Si plusieurs dimensions, privilégier la plus grande (dimension principale)
        if len(unique_dimensions) > 1:
            # Calculer la "taille" de chaque dimension pour prioriser
            def dimension_size(dim_str):
                try:
                    parts = dim_str.split('*')
                    nums = [float(p) for p in parts]
                    return sum(nums) * len(nums)  # Somme pondérée par le nombre de dimensions
                except:
                    return 0
            
            unique_dimensions.sort(key=dimension_size, reverse=True)
            return unique_dimensions[0]  # Retourner seulement la plus grande
        else:
            return unique_dimensions[0]
    
    return None

def extract_colors(text):
    """
    Extrait les couleurs du texte (exactement comme dans main_backup.py).
    Retourne une chaîne avec les couleurs trouvées, séparées par des virgules.
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
        "profond", "profonde", "profonds", "profondes",
        "canard"
    ]
    
    # Dictionnaire pour normaliser les adjectifs vers leur forme de base
    adjective_normalization = {
        "claire": "clair", "clairs": "clair", "claires": "clair",
        "foncée": "foncé", "foncés": "foncé", "foncées": "foncé",
        "pâle": "pâle", "pales": "pâle", "pâles": "pâle",
        "vive": "vif", "vifs": "vif", "vives": "vif",
        "douce": "doux", "douces": "doux",
        "légère": "léger", "légers": "léger", "légères": "léger",
        "profonde": "profond", "profonds": "profond", "profondes": "profond",
        "canard": "canard"
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

def predict_nature_original_file():
    """
    Utilise le modèle le plus récent pour prédire la Nature des produits
    dans le fichier original et compare avec les vraies valeurs
    """
    
    # Fichier à analyser
    input_file = "20210614 Ecommerce sales.xlsb"
    
    print("🔮 Prédiction de Nature avec le modèle entraîné")
    print("=" * 60)
    
    # Vérifier que le fichier existe
    if not os.path.exists(input_file):
        print(f"❌ Fichier non trouvé: {input_file}")
        print("📁 Fichiers disponibles:")
        for file in os.listdir("."):
            if file.endswith((".xlsx", ".xlsb")):
                print(f"   - {file}")
        return
    
    # Charger le modèle
    print("📚 Chargement du modèle...")
    loader = ModelLoader()
    
    models = loader.list_available_models()
    if not models:
        print("❌ Aucun modèle trouvé. Exécutez d'abord main.py pour créer un modèle.")
        return
    
    print(f"📊 Modèles disponibles: {len(models)}")
    for i, model in enumerate(models[:3], 1):  # Afficher les 3 plus récents
        print(f"   {i}. {model['timestamp']} - {model['type']} (score: {model['score']:.3f})")
    
    # Charger le modèle le plus récent
    if not loader.load_latest_model():
        print("❌ Impossible de charger le modèle")
        return
    
    model_info = loader.get_model_info()
    print(f"✅ Modèle chargé: {model_info['type']} du {model_info['timestamp']}")
    print(f"   📈 Score d'entraînement: {model_info['score']:.3f}")
    print(f"   🏷️  Nombre de classes: {model_info['classes']}")
    
    # Charger le fichier original
    print(f"\n📁 Chargement du fichier: {input_file}")
    try:
        df = pd.read_excel(input_file)
        print(f"✅ Fichier chargé: {len(df)} lignes")
        
        # Vérifier les colonnes nécessaires
        required_cols = ['Nature', 'Libellé produit']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Colonnes manquantes: {missing_cols}")
            print(f"📊 Colonnes disponibles: {list(df.columns)}")
            return
        
        # Nettoyer les données
        df_clean = df[['Nature', 'Libellé produit']].dropna()
        print(f"📊 Après nettoyage: {len(df_clean)} lignes valides")
        
        # Prédiction par lots pour économiser la mémoire
        batch_size = 5000
        total_rows = len(df_clean)
        predictions = []
        
        print(f"\n🔮 Prédiction par lots de {batch_size} éléments...")
        
        for i in range(0, total_rows, batch_size):
            batch_end = min(i + batch_size, total_rows)
            batch_data = df_clean['Libellé produit'].iloc[i:batch_end].tolist()
            
            # Prédiction du lot
            batch_predictions = loader.predict(batch_data)
            predictions.extend(batch_predictions)
            
            # Affichage du progrès
            progress = (batch_end / total_rows) * 100
            print(f"   Lot {i//batch_size + 1}: {progress:.1f}% terminé ({batch_end}/{total_rows})")
        
        # Ajouter les prédictions au DataFrame
        df_clean['predicted_nature'] = predictions
        
        # Extraire les couleurs et dimensions du libellé produit
        print(f"\n🎨 Extraction des couleurs et dimensions...")
        df_clean['couleurs_extraites'] = df_clean['Libellé produit'].apply(extract_colors)
        df_clean['dimensions_extraites'] = df_clean['Libellé produit'].apply(extract_dimensions)
        
        # Statistiques d'extraction
        colors_found = df_clean['couleurs_extraites'].str.len() > 0
        dimensions_found = df_clean['dimensions_extraites'].notna()
        
        print(f"   🎨 Couleurs trouvées: {colors_found.sum()}/{len(df_clean)} produits ({(colors_found.sum()/len(df_clean)*100):.1f}%)")
        print(f"   📏 Dimensions trouvées: {dimensions_found.sum()}/{len(df_clean)} produits ({(dimensions_found.sum()/len(df_clean)*100):.1f}%)")
        
        # Calculer la précision
        correct_predictions = (df_clean['Nature'] == df_clean['predicted_nature']).sum()
        total_predictions = len(df_clean)
        accuracy = (correct_predictions / total_predictions) * 100
        
        print(f"\n📊 Résultats de la prédiction:")
        print(f"   🎯 Précision globale: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
        print(f"   ❌ Erreurs: {total_predictions - correct_predictions} prédictions incorrectes")
        
        # Ajouter une colonne de validation
        df_clean['prediction_correcte'] = df_clean.apply(
            lambda row: 'VRAI' if row['Nature'] == row['predicted_nature'] else 'FAUX', axis=1
        )
        
        # Analyser les erreurs les plus fréquentes
        print(f"\n🔍 Analyse des erreurs:")
        errors = df_clean[df_clean['prediction_correcte'] == 'FAUX']
        
        if len(errors) > 0:
            # Top 10 des erreurs par catégorie réelle
            error_analysis = errors.groupby(['Nature', 'predicted_nature']).size().reset_index(name='count')
            error_analysis = error_analysis.sort_values('count', ascending=False)
            
            print(f"   📋 Top 10 des confusions les plus fréquentes:")
            for i, row in error_analysis.head(10).iterrows():
                print(f"      {row['count']:4d}x '{row['Nature']}' → '{row['predicted_nature']}'")
            
            # Catégories avec le plus d'erreurs
            print(f"\n   📊 Catégories avec le plus d'erreurs:")
            categories_errors = errors['Nature'].value_counts().head(10)
            for category, error_count in categories_errors.items():
                total_category = len(df_clean[df_clean['Nature'] == category])
                error_rate = (error_count / total_category) * 100
                print(f"      '{category}': {error_count}/{total_category} ({error_rate:.1f}% d'erreur)")
        
        # Analyser les prédictions par catégorie
        print(f"\n📈 Précision par catégorie (top 15):")
        category_stats = []
        
        for category in df_clean['Nature'].unique():
            category_data = df_clean[df_clean['Nature'] == category]
            category_correct = len(category_data[category_data['prediction_correcte'] == 'VRAI'])
            category_total = len(category_data)
            category_accuracy = (category_correct / category_total) * 100 if category_total > 0 else 0
            
            category_stats.append({
                'category': category,
                'correct': category_correct,
                'total': category_total,
                'accuracy': category_accuracy
            })
        
        # Trier par nombre total (plus représentatif)
        category_stats.sort(key=lambda x: x['total'], reverse=True)
        
        for stat in category_stats[:15]:
            print(f"   {stat['accuracy']:5.1f}% '{stat['category']}' ({stat['correct']}/{stat['total']})")
        
        # Exemples d'erreurs intéressantes
        print(f"\n📋 Exemples d'erreurs intéressantes:")
        interesting_errors = errors.sample(min(10, len(errors))) if len(errors) > 0 else pd.DataFrame()
        
        for i, row in interesting_errors.iterrows():
            libelle = row['Libellé produit'][:60] + "..." if len(row['Libellé produit']) > 60 else row['Libellé produit']
            print(f"   • '{libelle}'")
            print(f"     Nature originale: '{row['Nature']}' → Prédite: '{row['predicted_nature']}'")
        
        # Sauvegarder les résultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"predictions_original_file_{timestamp}.xlsx"
        
        # Préparer le DataFrame final avec toutes les colonnes
        df_final = df_clean.copy()
        
        # Ajouter les dimensions et couleurs si disponibles dans le fichier original
        if len(df) == len(df_final):
            # Ajouter toutes les autres colonnes du fichier original
            for col in df.columns:
                if col not in df_final.columns:
                    df_final[col] = df[col].iloc[:len(df_final)]
        
        df_final.to_excel(output_file, index=False)
        print(f"\n💾 Résultats sauvegardés: {output_file}")
        
        # Résumé final
        print(f"\n📊 Résumé final:")
        print(f"   📁 Fichier analysé: {input_file}")
        print(f"   🤖 Modèle utilisé: {model_info['type']} ({model_info['timestamp']})")
        print(f"   📈 Précision: {accuracy:.2f}%")
        print(f"   📊 Total produits: {total_predictions}")
        print(f"   ✅ Prédictions correctes: {correct_predictions}")
        print(f"   ❌ Prédictions incorrectes: {total_predictions - correct_predictions}")
        print(f"   🎨 Couleurs extraites: {colors_found.sum()} produits")
        print(f"   📏 Dimensions extraites: {dimensions_found.sum()} produits")
        print(f"   💾 Fichier de sortie: {output_file}")
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement: {str(e)}")
        import traceback
        traceback.print_exc()

def quick_sample_test():
    """Test rapide sur un échantillon de 1000 produits"""
    print("⚡ Test rapide sur échantillon")
    print("=" * 40)
    
    # Charger le modèle
    loader = ModelLoader()
    if not loader.load_latest_model():
        print("❌ Impossible de charger le modèle")
        return
    
    # Charger un échantillon
    df = pd.read_excel("20210614 Ecommerce sales.xlsb", nrows=1000)
    df_clean = df[['Nature', 'Libellé produit']].dropna()
    
    print(f"📊 Test sur {len(df_clean)} produits...")
    
    # Prédiction
    predictions = loader.predict(df_clean['Libellé produit'].tolist())
    accuracy = (df_clean['Nature'] == predictions).mean() * 100
    
    print(f"🎯 Précision sur l'échantillon: {accuracy:.2f}%")

if __name__ == "__main__":
    print("🔮 Script de prédiction Nature avec modèle entraîné")
    print("=" * 50)
    
    # Demander à l'utilisateur s'il veut un test rapide ou complet
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_sample_test()
    else:
        predict_nature_original_file()
    
    print("=" * 50)
    print("✅ Script terminé")
