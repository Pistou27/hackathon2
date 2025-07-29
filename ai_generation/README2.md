# Generative AI Content Pipeline with Workflow Automation

## ✨ Présentation du projet

Ce projet se divise en deux volets d'expérimentation autour de la génération de contenu par intelligence artificielle, avec une contrainte forte : être **exécutable sur CPU uniquement**, sans recours à des services cloud ou GPU.

- **Volet 1** : Prototype de génération de posts LinkedIn (texte, filtrage, scoring, image VAE)
- **Volet 2** : Pipeline complet et fonctionnel de création d’articles de blog (texte, résumé, image Stable Diffusion, automatisation)

---

## 🔹 Volet 1 — Génération de Posts LinkedIn (prototype)

### 🌟 Objectif

Créer un pipeline IA capable de :

- Générer du texte court à partir d’un prompt (distilGPT2)
- Appliquer un résumé (DistilBART)
- Mesurer la similarité sémantique (MiniLM)
- Filtrer les contenus sensibles (**filtre éthique** par regex)
- Générer une image simple (VAE)

### ⚠️ Limites

- Le pipeline est opérationnel mais les résultats sont limités (qualité moyenne du texte)
- VAE non lié au contenu textuel, simple preuve de concept

### 📊 Technologies

- `distilgpt2`, `sshleifer/distilbart-cnn-12-6`, `all-MiniLM-L6-v2`
- **Filtrage éthique** simple avec regex et mots-clés
- Modèle VAE entraîné sur MNIST

### 🌍 Exécution

```bash
# Installation
pip install -r requirements.txt

# Lancer la génération de posts
python automate.py

# Interface Streamlit (facultative)
streamlit run app.py
```

---

## 🔹 Volet 2 — Pipeline de Blog (fonctionnel)

### 🌟 Objectif

Générer automatiquement un article de blog complet :

- Texte via `LaMini-Flan-T5`
- Résumé automatique (DistilBART)
- Scoring sémantique (MiniLM)
- **Filtrage éthique avancé** (`better-profanity`)
- Image avec Stable Diffusion
- Article final sauvegardé en Markdown + JSON

### ✅ Fonctionnel & Automatisé

Pipeline complet prêt à l’emploi avec planification possible (génération toutes les heures, par exemple).

### 📊 Technologies principales

- `MBZUAI/LaMini-Flan-T5-783M`, `distilbart-cnn-12-6`, `MiniLM`
- `diffusers`, `better-profanity`, `schedule`, `streamlit`

### 🌍 Exécution

```bash
# Installation
cd new_test
pip install -r requirements.txt

# Lancer un article de blog
python pipeline.py "Sujet de l'article"

# Interface de lecture
streamlit run blog_viewer.py

# Planification automatique
python run_scheduler.py
```

---

## 🔍 Arborescence simplifiée

```
ai_generation/
├── app.py                  # UI Streamlit (posts)
├── automate.py             # Pipeline LinkedIn
├── vae_images/             # Expérimentations VAE
├── gpt2_lora_finetuned/    # Fine-tuning LoRA (non intégré)
├── new_test/               # Pipeline blog (fonctionnel)
│   ├── pipeline.py         # Script principal
│   ├── blog_viewer.py      # Interface articles
│   ├── outputs/            # Articles générés
│   └── run_scheduler.py    # Automatisation
```

---

## 🚫 Limitations connues

- Tout tourne sur CPU → lenteur sur les grandes générations
- Stable Diffusion économe, mais long sur CPU
- Contenu parfois générique
- **Filtrage éthique basique dans le volet 1**, mais **filtrage amélioré dans le volet 2** (`better-profanity`)

---

## 💼 Auteurs & licence

Projet réalisé dans le cadre d'un hackathon par :

- **Pistou27** ([https://github.com/Pistou27](https://github.com/Pistou27))

Les autres contributeurs apparaissent dans l'historique du repo.

⚡ Licence non spécifiée : ce projet n'est pas explicitement libre de droits.

---

## 📄 Pour aller plus loin

- Ajouter un classifieur éthique entraîné localement
- Connecter à une API de publication (LinkedIn, blog...)
- Optimiser les modèles pour CPU (Quantization, ONNX...)
- Créer une base de thèmes aléatoires pour la planification automatique

---

© Hackathon2 - 2025

