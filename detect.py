import os
import pickle
import numpy as np
import joblib
# --- Charger tous les modèles GMM ---
def load_models(models_folder="models"):
    languages = ['Arabic', 'French', 'German', 'Japanese', 'Spanish']
    models = {}
    for lang in languages:
        models[lang] = []
        lang_folder = os.path.join(models_folder, lang)
        # Boucle sur tous les sous-dossiers GMM_4, GMM_8, etc.
        for gmm_sub in os.listdir(lang_folder):
            gmm_path = os.path.join(lang_folder, gmm_sub, "model.joblib")
            if os.path.exists(gmm_path):
                models[lang].append(joblib.load(gmm_path))
    return models

# --- Calcul score moyen d'une langue ---
def score_language(features, gmm_list):
    scores = [gmm.score(features) for gmm in gmm_list]
    return np.mean(scores)

# --- Identifier langue ---
def detect_language(features, models):
    language_scores = {lang: score_language(features, models[lang]) for lang in models}
    detected_language = max(language_scores, key=language_scores.get)
    return detected_language, language_scores
 