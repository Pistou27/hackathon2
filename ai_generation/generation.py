from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class TextGenerator:
    """
    Classe de génération de texte avec GPT2-medium
    """

    def __init__(self, model_name="gpt2-medium", max_length=200):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.max_length = max_length  # nombre de tokens à générer

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            repetition_penalty=1.2,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
