from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, textwrap

class ArticleGenerator:
    """
    Génère un article court (≤300 mots) sur un sujet.
    Par défaut : beam-search déterministe → moins d’hallucinations.
    """

    def __init__(self, model_id: str = "distilgpt2", max_new: int = 380):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model     = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()
        self.max_new   = max_new

    def generate(self, topic: str, temperature: float = 0.35) -> str:
        """
        topic       : sujet de l'article
        temperature : 0.2-0.6 recommandé (créativité modérée)
        """
        prompt = (
            "Rédige un article clair et structuré (max 300 mots).\n"
            "Sujet : " + topic + "\n"
            "Tonalité : informative, synthétique.\n"
            "Évite le code, les digressions hors-sujet et les anglicismes.\n\n"
            "Article :\n"
        )

        ids = self.tokenizer(prompt, return_tensors="pt")

        out = self.model.generate(
            **ids,
            max_new_tokens=self.max_new,
            do_sample=False,          # beam search déterministe
            num_beams=4,
            temperature=temperature,  # influe peu car do_sample=False
            repetition_penalty=1.20,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # on retire le prompt
        return "\n".join(text.splitlines()[len(prompt.splitlines()):]).strip()