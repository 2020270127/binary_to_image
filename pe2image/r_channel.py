import math
import numpy as np
import torch
import torch.nn.functional as F

def optimized_entropy_r_channel(data: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
    n = data.shape[0]
    if window_size < 2 or n < window_size:
        return np.array([], dtype=np.uint8)
    
    max_bits = math.log2(window_size) if window_size < 256 else 8.0
    scale = 255.0 / max_bits
    entropies = []
    
    if step == 1:
        log2_table = np.zeros(window_size + 1, dtype=np.float32)
        for i in range(1, window_size + 1):
            log2_table[i] = math.log2(i)
        freq = np.zeros(256, dtype=np.int32)
        for b in data[:window_size]:
            freq[b] += 1
        S = 0.0
        for count in freq:
            if count > 0:
                S += count * log2_table[count]
        H = math.log2(window_size) - S / window_size
        entropies.append(int(H * scale))
        
        for i in range(1, n - window_size + 1):
            old_byte = data[i - 1]
            new_byte = data[i + window_size - 1]
            if old_byte == new_byte:
                entropies.append(entropies[-1])
                continue
            old_count = freq[old_byte] 
            new_count = freq[new_byte] 
            freq[old_byte] -= 1 
            freq[new_byte] += 1
            if old_count > 0:
                S -= old_count * log2_table[old_count] 
            if new_count > 0:
                S -= new_count * log2_table[new_count] 
            new_old = old_count - 1 
            new_new = new_count + 1
            if new_old > 0:
                S += new_old * log2_table[new_old]
            if new_new > 0:
                S += new_new * log2_table[new_new]
            H = math.log2(window_size) - S / window_size
            entropies.append(int(H * scale))
    else:
        for i in range(0, n - window_size + 1, step):
            window = data[i:i+window_size]
            counts = np.bincount(window, minlength=256)
            probs = counts / window_size
            H = 0.0
            for p in probs:
                if p > 0:
                    H -= p * math.log2(p)
            entropies.append(int(H * scale))
    
    return np.array(entropies, dtype=np.uint8)

def scale_entropy_map(ent_map: np.ndarray, mode: str = "threshold7_5_exp1_2") -> np.ndarray:
    flat = ent_map.ravel().astype(np.float32)
    bits = (flat / 255.0) * 8.0  
    scaled = np.zeros_like(bits)

    if mode == "threshold6_quad":
        adj = np.maximum(0.0, bits - 6.0) / 2.0
        scaled = (adj ** 2.0) * 255.0

    elif mode == "threshold7_5_soft":
        adj = np.maximum(0.0, bits - 7.5) / 0.5  
        scaled_f = adj * 255.0
        scaled = scaled_f
    elif mode == "threshold7_5_exp1_2":
        adj = np.maximum(0.0, bits - 7.5) / 0.5
        scaled_f = (adj ** 1.2) * 255.0
        scaled = scaled_f
    else:
        scaled_f = ((bits / 8.0) ** 2.0) * 255.0
        scaled = scaled_f
    out = np.clip(scaled, 0, 255).astype(np.uint8)
    return out.reshape(ent_map.shape)

def apply_entropy_to_r(pe_image, step: int = 1, scaling_mode: str = "threshold7_5_exp1_2") -> None:
    if pe_image.pe_map.ndim != 1:
        flat_data = pe_image.pe_map.flatten()
    else:
        flat_data = pe_image.pe_map

    entropy_vector = optimized_entropy_r_channel(flat_data, window_size=pe_image.window_size, step=step)
    
    H = pe_image.pe_image_height
    W = pe_image.pe_image_width
    small_h = (H - pe_image.window_size) // step + 1
    small_w = (W - pe_image.window_size) // step + 1  

    required_size = small_h * small_w
    current_size = entropy_vector.shape[0]

    if current_size < required_size:
        entropy_vector = np.pad(entropy_vector, (0, required_size - current_size), mode='constant')
    elif current_size > required_size:
        entropy_vector = entropy_vector[:required_size]

    small_entropy_map = entropy_vector.reshape((small_h, small_w)).astype(np.float32)

    
    if current_size < required_size:
        entropy_vector = np.pad(entropy_vector, (0, required_size - current_size), mode='constant')
    elif current_size > required_size:
        entropy_vector = entropy_vector[:required_size]
    small_entropy_map = entropy_vector.reshape((small_h, small_w)).astype(np.float32)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor_small = torch.from_numpy(small_entropy_map).unsqueeze(0).unsqueeze(0).to(device)
    tensor_up = F.interpolate(tensor_small, size=(H, W), mode='bilinear', align_corners=False)
    upsampled_map = tensor_up.squeeze(0).squeeze(0).cpu().numpy()
    
    scaled_map = scale_entropy_map(upsampled_map, mode=scaling_mode)
    flatten_scaled = scaled_map.ravel()
    length = min(pe_image.pe_rgb_map.shape[0], flatten_scaled.shape[0])
    pe_image.pe_rgb_map[:length, 0] = flatten_scaled[:length]
