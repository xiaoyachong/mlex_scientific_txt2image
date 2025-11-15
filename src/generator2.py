import os
# from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
import torch

device = "cuda"

# load model
model_path = "./sd-naruto-model-lora-sdxl-dis1000_b/" #pytorch_lora_weights.bin"
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     torch_dtype=torch.float16,
#     safety_checker=None,
#     feature_extractor=None,
#     requires_safety_checker=False
# )
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
# load lora weights
pipe.unet.load_attn_procs(model_path)
# set to use GPU for inference
pipe.to(device)

# generate image
prompt = '5 nm raduis Sphere revealed by GISAXS data'

prompt = prompt
out_dir = '10_generated_sdxl_dis1000_b/{}'.format(prompt)
out_name = 'sphere_5nm'

if not os.path.exists(f'./{out_dir}'):
    os.makedirs(f'./{out_dir}')

#save image
for i in range(50):
    print(f"Generating image {i}")
    image = pipe(prompt, num_inference_steps=30).images[0]
    image.save(f"./{out_dir}/{out_name}_{i}.jpg")
    
    
