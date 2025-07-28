from diffusers import StableDiffusionPipeline
import torch, os

class ImageGenerator:
    def __init__(self, model_id="stabilityai/sd-turbo"):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float32)
        self.pipe.to("cpu")

    def generate(self, prompt:str, path="generated.png"):
        image = self.pipe(prompt, num_inference_steps=15).images[0]
        image.save(path)
        return path