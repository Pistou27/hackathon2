from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel, PeftConfig
import torch

# 1. Charger le tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_lora_finetuned")
tokenizer.pad_token = tokenizer.eos_token  # utile pour éviter des erreurs de padding

# 2. Charger la config LoRA
peft_config = PeftConfig.from_pretrained("./gpt2_lora_finetuned")

# 3. Charger le modèle de base GPT-2
base_model = GPT2LMHeadModel.from_pretrained(peft_config.base_model_name_or_path)

# 4. Appliquer les poids LoRA fine-tunés
model = PeftModel.from_pretrained(base_model, "./gpt2_lora_finetuned")

# 5. Passage en mode évaluation
model.eval()

# 6. Prompt utilisateur
prompt = "Write a LinkedIn post about the future of artificial intelligence:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 7. Génération du texte
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

# 8. Affichage du résultat
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n📝 Post généré :\n")
print(generated_text)
