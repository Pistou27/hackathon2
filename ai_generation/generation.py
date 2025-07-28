from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger

# --------------------------------------------------------------------------- #
# 1. Dépendance optionnelle : PEFT (LoRA)
# --------------------------------------------------------------------------- #
try:
    from peft import PeftModel, PeftConfig
    _PEFT_OK = True
except ImportError:
    _PEFT_OK = False

# 2. Dépendance optionnelle : bitsandbytes 8-bit
try:
    from transformers import BitsAndBytesConfig
    _BNB_OK = True
except ImportError:
    _BNB_OK = False


class TextGenerator:
    """
    Générateur unifié GPT-like (GPT-2 / GPT-Neo …) avec support LoRA (PEFT)
    et quantization 8-bit (GPU uniquement).

    • Sur GPU  : option quantize_8bit=True active BitsAndBytes.
    • Sur CPU  : forçage en FP32 (quantize_8bit et dtype ignorés).
    """

    def __init__(
        self,
        model_path: str = "gpt2-medium",
        max_new_tokens: int = 200,
        quantize_8bit: bool = True,
        dtype: torch.dtype | None = torch.float16,
        device: str | None = None,
    ):
        # ------------------ contexte matériel --------------------------------
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        use_gpu = self.device.type == "cuda"

        # Sur CPU : on désactive 8-bit + FP16 pour éviter NaN/Inf
        if not use_gpu:
            quantize_8bit = False
            dtype = None
            logger.info("🖥️  CPU détecté ➜ désactivation quantization 8-bit & FP16")

        self.max_new_tokens = max_new_tokens

        # ------------------ tokenizer ----------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # ------------------ arguments de chargement --------------------------
        load_kwargs = {}

        # >>> Mode sûr : pas de 8-bit, FP32 si GPU, sinon défaut CPU
        quantize_8bit = False
        dtype = torch.float32 if self.device.type == "cuda" else None
        logger.info(f"🔒 Mode sûr: 8bit désactivé, dtype={dtype}")

        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype

        # ------------------ modèle ou adaptateur LoRA ------------------------
        lora_cfg_file = Path(model_path, "adapter_config.json")
        if lora_cfg_file.exists():
            if not _PEFT_OK:
                raise ImportError("Le dossier contient un adaptateur LoRA, "
                                  "mais le package 'peft' n'est pas installé. "
                                  "👉  pip install peft")
            logger.info("⚡️ Adaptateur LoRA détecté → chargement PEFT")
            peft_cfg = PeftConfig.from_pretrained(model_path)
            base = AutoModelForCausalLM.from_pretrained(peft_cfg.base_model_name_or_path, **load_kwargs)
            self.model = PeftModel.from_pretrained(base, model_path, is_trainable=False)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        # ------------------ placement device (sauf modèle 8-bit) -------------
        if not getattr(self.model, "is_loaded_in_8bit", False):
            self.model.to(self.device)

        self.model.eval()
        torch.set_grad_enabled(False)

    # ----------------------------------------------------------------------- #
    def generate(self, prompt: str, **gen_kwargs) -> str:
        """
        Génère du texte à partir d'un prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

# --------------------------------------------------------------------------- #
# Usage CLI rapide
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="*", help="Texte du prompt")
    parser.add_argument("--model", default="gpt2-medium",
                        help="Chemin HF ou dossier LoRA")
    parser.add_argument("--no-8bit", action="store_true",
                        help="Force le chargement en FP32 même sur GPU")
    args = parser.parse_args()

    gen = TextGenerator(model_path=args.model,
                        quantize_8bit=not args.no_8bit)
    prompt_text = " ".join(args.prompt) or input("Prompt: ")
    print("\n" + "-"*60)
    print(gen.generate(prompt_text))

def has_adapter(self) -> bool:
        return hasattr(self.model, "disable_adapter")

def set_adapter_state(self, enabled: bool):
    """
    True  → active l’adaptateur LoRA
    False → désactive (retour au GPT-2 vanilla)
    """
    if not self.has_adapter():
        return
    if enabled:
        self.model.enable_adapter()
    else:
        self.model.disable_adapter()