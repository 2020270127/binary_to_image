import numpy as np
import torch
import torch.nn.functional as F

def optimized_string_ratio_1d(data: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
    printable = ((data >= 32) & (data <= 126)).astype(np.uint8)
    kernel = np.ones(window_size, dtype=np.uint8)
    conv_result = np.convolve(printable, kernel, mode='valid')
    ratio = conv_result / window_size
    return ratio[::step]

def apply_string_ratio_to_g(pe_image, step: int = 1) -> None:
    if pe_image.pe_map.ndim != 1:
        flat_data = pe_image.pe_map.flatten()
    else:
        flat_data = pe_image.pe_map

    ratio_vector = optimized_string_ratio_1d(flat_data, window_size=pe_image.window_size, step=step)
    
    H = pe_image.pe_image_height
    W = pe_image.pe_image_width
    small_h = (H - pe_image.window_size) // step + 1
    small_w = (W - pe_image.window_size) // step + 1
    required_size = small_h * small_w
    current_size = ratio_vector.shape[0]
    if current_size < required_size:
        ratio_vector = np.pad(ratio_vector, (0, required_size - current_size), mode='constant')
    elif current_size > required_size:
        ratio_vector = ratio_vector[:required_size]
    small_ratio_map = ratio_vector.reshape((small_h, small_w))
    
    small_ratio_map_np = small_ratio_map.astype(np.float32) ** 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor_small = torch.from_numpy(small_ratio_map_np).unsqueeze(0).unsqueeze(0).to(device)
    tensor_up = F.interpolate(tensor_small, size=(H, W), mode='bilinear', align_corners=False)
    upsampled_map = tensor_up.squeeze(0).squeeze(0).cpu().numpy()
    
    g_val = np.clip(upsampled_map * 255, 0, 255).astype(np.uint8)
    pe_image.pe_rgb_map[:, 1] = g_val.flatten()
