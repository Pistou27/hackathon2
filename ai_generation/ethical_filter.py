from transformers import pipeline

# Chargement global du classificateur
bert_classifier = pipeline("text-classification", model="unitary/toxic-bert", truncation=True)

def ethical_filter(text: str, threshold: float = 0.5) -> dict:
    """
    Analyse le texte avec BERT (unitary/toxic-bert) pour détecter la toxicité.

    Args:
        text (str): Texte à analyser.
        threshold (float): Seuil de détection.

    Returns:
        dict: Détails sur la toxicité détectée.
    """
    result = {
        "flagged": False,
        "status": "OK",
        "label": None,
        "score": 0.0
    }

    try:
        prediction = bert_classifier(text)[0]
        label = prediction["label"].lower()
        score = prediction["score"]

        result["label"] = label
        result["score"] = score
        result["flagged"] = (label == "toxic" and score >= threshold)

        if result["flagged"]:
            result["status"] = "Toxique détecté"
    except Exception as e:
        result["status"] = f"Erreur BERT : {e}"
        result["flagged"] = False

    return result