from diffusion.unets.unet_2d_condition import UNet2DConditionModel
from diffusion.pipeline import StableDiffusionPipeline
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from transformers import AutoTokenizer, PretrainedConfig
import torch


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    model_class = text_encoder_config.architectures[0]

    assert model_class == "CLIPTextModel", f"model_class should be CLIPTextModel"

    from transformers import CLIPTextModel

    return CLIPTextModel


model_id = "stabilityai/stable-diffusion-2-1-base"

# import correct text encoder class
text_encoder_cls = import_model_class_from_model_name_or_path(model_id)

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
text_encoder = text_encoder_cls.from_pretrained(model_id, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    subfolder="tokenizer",
    use_fast=False,
)


pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    # controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float32,
)
pipe = pipeline.to("cuda")

from safetensors import safe_open

tensors1={}
# light embedding at weight
with safe_open("runs/test/checkpoint-2000/model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors1[key] = f.get_tensor(key)

tensors2={}
# light embedding at weight
with safe_open("runs/test/checkpoint-2000/model_1.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors2[key] = f.get_tensor(key)

clip_embedding_size = 1024
num_camera_type = 11
num_lighting_type = 14
light_encoder = torch.nn.Embedding(num_lighting_type, clip_embedding_size)
camera_pose_encoder = torch.nn.Embedding(num_camera_type, clip_embedding_size)


light_type = torch.full((1,), 1, dtype=torch.long)
camera_encoder_type = torch.full((1,), 1, dtype=torch.long)
light_encoder.load_state_dict(tensors1)
camera_pose_encoder.load_state_dict(tensors2)

embedding = light_encoder(light_type) + camera_pose_encoder(camera_encoder_type)

prompt = "car"
image = pipe(prompt, embedding=embedding).images[0]

image.save("test1.png")
