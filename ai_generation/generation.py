from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


class TextGenerator:
    """
    Classe de génération de texte avec distilGPT2
    """

    def __init__(self, model_name="distilgpt2", max_length=100):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.max_length = max_length

    def generate(self, prompt: str) -> str:
        """
        Génère un texte à partir d'un prompt.

        Args:
            prompt (str): Texte d'entrée

        Returns:
            str: Texte généré
        """

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(
            input_ids,
            max_length=self.max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            num_return_sequences=1
        )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text