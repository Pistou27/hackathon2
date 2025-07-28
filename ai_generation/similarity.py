from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SimilarityChecker:
    """
    Vérifie la similarité sémantique entre prompt et résumé
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def compare(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité cosinus entre deux textes

        Args:
            text1 (str): premier texte (ex. prompt)
            text2 (str): second texte (ex. résumé)

        Returns:
            float: score de similarité entre 0 et 1
        """
        embeddings = self.model.encode([text1, text2])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return round(score, 4)  # 4 décimales pour clarté