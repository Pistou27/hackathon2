from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, textwrap

class ArticleGenerator:
    def __init__(self, model_id="distilgpt2", max_new=400):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()
        self.max_new = max_new

    def generate(self, topic: str, temperature: float = 0.8) -> str:
        """
        Génère un article ≈300 mots sur *topic*.
        temperature: 0.2-1.2 (plus bas = plus factuel, plus haut = plus créatif)
        """
        prompt = f"Écris un article de blog (300 mots) sur : {topic}\n\n"
        ids = self.tokenizer(prompt, return_tensors="pt")

        out = self.model.generate(
            **ids,
            max_new_tokens=self.max_new,
            do_sample=True,
            top_p=0.92,
            top_k=40,
            temperature=temperature,      # ← on utilise la valeur passée
            repetition_penalty=1.15,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return "\n".join(text.splitlines()[1:]).strip()