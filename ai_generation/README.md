Générative AI Content Pipeline (CPU-Friendly)

# 🚀 Generative AI Content Pipeline (Hackathon)

## 📌 Objectif

Ce projet propose un pipeline IA modulaire pour la **génération automatique de texte**, le **contrôle qualité**, le **filtrage éthique** et la **génération d'images (optionnelle)** — le tout optimisé pour **fonctionner sur CPU**, sans API commerciale.

---

## 🛠️ Pipeline complet

Prompt utilisateur
↓
distilGPT2 — génération
↓
distilBART — résumé automatique
↓
MiniLM — similarité résumé/prompt
↓
Filtre éthique (regex/keywords)
↓
Texte final tagué ou rejeté


---

## 📦 Structure du projet

.
├── data_loader.py # Chargement du dataset IMDB
├── generation.py # Génération de texte (distilGPT2)
├── summarization.py # Résumé automatique (distilBART)
├── similarity.py # Vérification de cohérence (MiniLM)
├── ethical_filter.py # Détection de dérives éthiques
├── evaluation.py # BLEU / ROUGE / Robustesse
├── automate.py # Pipeline complet (CLI)
├── automate_schedule.py # Version planifiée avec schedule
├── vae_model.py # VAE image (optionnel)
├── train_vae.py # Entraînement sur MNIST
├── generate_image.py # Génération aléatoire d’image
├── main.py # Test manuel (5 exemples)
└── adversarial_prompts.txt # Prompts bruités pour robustesse


---

## 🔧 Installation

```bash
pip install -r requirements.txt

Modèles utilisés :

    distilgpt2

    sshleifer/distilbart-cnn-12-6

    sentence-transformers/all-MiniLM-L6-v2

🚦 Lancer un test complet

python automate.py

Pour planifier l’exécution régulière :

python automate_schedule.py

Pour tester des prompts absurdes :

from evaluation import evaluate_adversarial
evaluate_adversarial()

📊 Évaluation

    BLEU moyen : ~0.25

    ROUGE-L moyen : ~0.45

    Similarité prompt/résumé : ~0.72

    Taux de détection éthique : ~5% (mots-clés sensibles)

🧠 Technologies

    transformers (Hugging Face)

    sentence-transformers

    nltk, rouge-score, scikit-learn

    torch, matplotlib (pour VAE)

    schedule (planification)

🧩 À améliorer

    Ajout d’un classifieur de toxicité entraîné localement

    Passage vers interface Streamlit

    Élargissement au multilingue