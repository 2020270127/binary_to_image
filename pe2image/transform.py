import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from . import utils

def transform_to_fixed_size_interpolation(pe_image, output_size: tuple = (256, 256)) -> np.ndarray:
    input_image = utils.reshape_as_image(pe_image.pe_rgb_map, pe_image.pe_image_height, pe_image.pe_image_width)
    img_tensor = torch.from_numpy(input_image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    resized_tensor = F.interpolate(img_tensor, size=output_size, mode='bilinear', align_corners=False) # interpolation -> bilinear
    resized_tensor = (resized_tensor * 255.0).clamp(0, 255).byte()
    resized_img = resized_tensor.squeeze(0).permute(1, 2, 0).numpy()
    return resized_img.astype(np.uint8)

def save_fixed_interpolation(pe_image, base_output_dir: str, file_name: str, output_size: tuple = (256, 256)) -> None:
    rgb_folder = os.path.join(base_output_dir, "RGB")
    os.makedirs(rgb_folder, exist_ok=True)
    fixed_img = transform_to_fixed_size_interpolation(pe_image, output_size=output_size)
    Image.fromarray(fixed_img, mode='RGB').save(
        os.path.join(rgb_folder, f"{file_name}.png")
    )

def save_channel_color_images(pe_image, base_output_dir: str, file_name: str, output_size: tuple = (256, 256)) -> None:
    fixed_img = transform_to_fixed_size_interpolation(pe_image, output_size=output_size)
    r_folder = os.path.join(base_output_dir, "R")
    g_folder = os.path.join(base_output_dir, "G")
    b_folder = os.path.join(base_output_dir, "B")
    os.makedirs(r_folder, exist_ok=True)
    os.makedirs(g_folder, exist_ok=True)
    os.makedirs(b_folder, exist_ok=True)

    r_img = np.zeros_like(fixed_img)
    g_img = np.zeros_like(fixed_img)
    b_img = np.zeros_like(fixed_img)
    r_img[..., 0] = fixed_img[..., 0]
    g_img[..., 1] = fixed_img[..., 1]
    b_img[..., 2] = fixed_img[..., 2]

    Image.fromarray(r_img, mode='RGB').save(os.path.join(r_folder, f"{file_name}_R.png"))
    Image.fromarray(g_img, mode='RGB').save(os.path.join(g_folder, f"{file_name}_G.png"))
    Image.fromarray(b_img, mode='RGB').save(os.path.join(b_folder, f"{file_name}_B.png"))

def save_original_image(pe_image, base_output_dir: str, file_name: str) -> None:
    """save original image before interpolation"""
    orig_folder = os.path.join(base_output_dir, "Original")
    os.makedirs(orig_folder, exist_ok=True)
    original_img_array = utils.reshape_as_image(pe_image.pe_rgb_map, pe_image.pe_image_height, pe_image.pe_image_width)
    Image.fromarray(original_img_array, mode='RGB').save(
        os.path.join(orig_folder, f"{file_name}_original_{pe_image.pe_image_width}x{pe_image.pe_image_height}.png")
    )
