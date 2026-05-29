import os
import torch
import sys
import folder_paths

from PIL import Image
from ..utils import rgba_to_rgb, simple_remove
from ..custom_models.utils import load_pipeline
from ...scripts.utils import rotate_normals_torch
from ...scripts.all_typing import *

comfy_path = os.path.dirname(folder_paths.__file__)

unique3d_path = f'{comfy_path}/custom_nodes/ComfyUI-Unique3D'
unique3d_ckpt_path = f'{comfy_path}/models/unique3d/ckpt'

sys.path.append(unique3d_path)

training_config = f"{unique3d_path}/app/custom_models/image2normal.yaml"
checkpoint_path = f"{unique3d_ckpt_path}/image2normal/unet_state_dict.pth"

trainer, pipeline = load_pipeline(training_config, checkpoint_path)
pipeline.enable_vae_slicing()
try:
    pipeline.enable_attention_slicing()
except AttributeError:
    pass
# pipeline.enable_model_cpu_offload()

def predict_normals(image: List[Image.Image], guidance_scale=2., do_rotate=True, num_inference_steps=30, **kwargs):
    img_list = image if isinstance(image, list) else [image]
    img_list = [rgba_to_rgb(i) if i.mode == 'RGBA' else i for i in img_list]
    images = []
    for img in img_list:
        pred_images = trainer.pipeline_forward(
            pipeline=pipeline,
            image=img,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        ).images
        images.extend(pred_images)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    images = simple_remove(images)
    if do_rotate and len(images) > 1:
        images = rotate_normals_torch(images, return_types='pil')
    return images