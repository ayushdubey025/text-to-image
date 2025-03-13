!pip install --upgrade diffusers transformers -q

from diffusers import StableDiffusionPipeline
import torch

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator().manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2-1"
    image_gen_size = (400,400)
    image_gen_guidance_scale = 9

HUGGINGFACE_TOKEN = "your_hugging_face_auth_token"  # Replace with your token

image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token=HUGGINGFACE_TOKEN  # Use the token here
).to(CFG.device)

def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    
    image = image.resize(CFG.image_gen_size)
    return image

# Test the function
generate_image("astronaut in space", image_gen_model)
