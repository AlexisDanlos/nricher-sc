from load_model import ModelLoader

print("🧪 Test de chargement et prédiction de modèle")
print("=" * 50)

# Load and use the latest model
loader = ModelLoader()

# Lister les modèles disponibles
print("📚 Modèles disponibles:")
models = loader.list_available_models()

if not models:
    print("❌ Aucun modèle trouvé. Exécutez d'abord main.py pour créer un modèle.")
    exit()

for i, model in enumerate(models, 1):
    print(f"   {i}. {model['timestamp']} - {model['type']} (score: {model['score']:.3f})")

print(f"\n🔄 Chargement du modèle le plus récent...")
if loader.load_latest_model():
    info = loader.get_model_info()
    print(f"✅ Modèle chargé: {info['type']} du {info['timestamp']}")
    print(f"   📊 Score: {info['score']:.3f}")
    print(f"   🏷️  Classes: {info['classes']}")
    print(f"   💻 Device: {info['device']}")
    
    # Make predictions
    test_products = [
        "Table en bois de chêne 120x80 cm",
        "Chaise ergonomique noire en cuir",
        "Lampe de bureau LED blanche",
        "Canapé 3 places gris anthracite",
        "Étagère murale 5 niveaux bois",
        "Bureau en verre avec tiroirs",
        "Tapis moderne 160x230 cm",
        "Meuble TV en chêne massif",
        "Baladeur MP3 étanche avec Bluetooth",
        "Enceinte portable étanche 20W",
        "Couette en soie 240x220 cm",
        "Oreiller mémoire de forme ergonomique",
        "Étagère de cuisine en métal noir",
        "Cafetière italienne en inox 6 tasses",
        "Gros caca de chien dans le jardin",
    ]
    
    print(f"\n🎯 Test de prédictions sur {len(test_products)} produits:")
    print("-" * 60)
    
    predictions = loader.predict(test_products)
    
    if predictions is not None:
        for i, (product, prediction) in enumerate(zip(test_products, predictions), 1):
            print(f"{i:2d}. '{product}'")
            print(f"    → Catégorie prédite: {prediction}")
            print()
        
        print("✅ Test terminé avec succès!")
    else:
        print("❌ Erreur lors des prédictions")
else:
    print("❌ Impossible de charger le modèle")