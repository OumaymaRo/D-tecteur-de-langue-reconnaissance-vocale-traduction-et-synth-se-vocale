import os
from features import extract_mfcc_from_bytes  # ta fonction MFCC
from detect import load_models, detect_language

# --- Chemin vers le fichier audio à tester ---
test_audio_path = r"C:\Users\LENOVO\Downloads\test.wav" # Remplace par ton fichier .wav

# --- Charger les modèles GMM ---
models = load_models("models")

# --- Extraire MFCC depuis le fichier ---
with open(test_audio_path, "rb") as f:
    audio_bytes = f.read()

features = extract_mfcc_from_bytes(audio_bytes)

# --- Détection de langue ---
detected_language, language_scores = detect_language(features, models)

# --- Afficher les résultats ---
print(f"Langue détectée : {detected_language}")
print("Scores par langue :")
for lang, score in language_scores.items():
    print(f"{lang}: {score:.2f}")
