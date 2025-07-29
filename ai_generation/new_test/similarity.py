from sentence_transformers import CrossEncoder

class SimilarityChecker:
    def __init__(self, model_id="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_id)

    def compare(self, doc: str, query: str) -> float:
        # Cross-encoder donne directement un score 0-1
        return float(self.model.predict([(query, doc)])[0])