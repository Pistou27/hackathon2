from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class ArticleGenerator:
    def __init__(self, model_id="MBZUAI/LaMini-Flan-T5-783M", max_new=380):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.model.eval()
        self.max_new = max_new

    def generate(self, topic: str) -> str:
        prompt = (
            f"Écris un article de blog structuré (250 à 300 mots) sur le thème suivant : {topic}.\n\n"
            "Structure : Titre, introduction, trois parties distinctes, conclusion.\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.2,
        )
        generated = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Enlever le prompt initial de la sortie
        return generated.strip()