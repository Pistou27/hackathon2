from transformers import BartTokenizer, BartForConditionalGeneration
import torch


class TextSummarizer:
    """
    Résumeur automatique basé sur distilBART
    """

    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6", max_input_length=512, max_summary_length=80):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.max_input_length = max_input_length
        self.max_summary_length = max_summary_length

    def summarize(self, text: str) -> str:
        """
        Résume un texte donné

        Args:
            text (str): Texte à résumer

        Returns:
            str: Résumé
        """

        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=self.max_summary_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary