from diffusers import StableDiffusionPipeline
import torch

class ImageGenerator:
    """
    Génère une image 512×512 avec SD-Turbo (CPU friendly).
    """

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", steps: int = 15):
        self.pipe  = StableDiffusionPipeline.from_pretrained(
            model_id,
            safety_checker=None,
            torch_dtype=torch.float32
        ).to("cpu")
        self.steps = steps

    def generate(
        self,
        prompt: str,
        negative: str | None = None,
        path: str = "generated.png",
    ) -> str:
        """
        prompt   : prompt positif détaillé
        negative : ce qu’on veut éviter (text, watermark…)
        path     : chemin de sauvegarde
        """
        result = self.pipe(
            prompt,
            negative_prompt=negative,
            num_inference_steps=self.steps
        )
        result.images[0].save(path)
        return path