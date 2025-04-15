import numpy as np
import math
from scipy.stats import entropy
from scipy.ndimage import uniform_filter
from tqdm import tqdm
import torch

def window_avg(data_2d: list, i: int, j: int, window_size: int = 3) -> int:
    half = window_size // 2
    total = 0
    count = 0
    for x in range(i - half, i + half + 1):
        if x < 0 or x >= len(data_2d):
            continue
        for y in range(j - half, j + half + 1):
            if y < 0 or y >= len(data_2d[x]):
                continue
            total += data_2d[x][y]
            count += 1
    return total // count if count > 0 else 0

def compute_average_map(data_2d: list, window_size: int = 3, verbose: bool = False) -> list:
    height = len(data_2d)
    width = len(data_2d[0])
    avg_map = [[0.0 for _ in range(width)] for _ in range(height)]
    for i in tqdm(range(height), desc="Computing average map", disable=not verbose):
        for j in range(width):
            avg_map[i][j] = window_avg(data_2d, i, j, window_size)
    return avg_map

def compute_average_map_np(data_2d: np.ndarray, window_size: int = 3) -> np.ndarray:
    if torch.cuda.is_available():
        device = 'cuda'
        kernel = torch.ones((1, 1, window_size, window_size), device=device) / (window_size * window_size)
        data_tensor = torch.from_numpy(data_2d).float().unsqueeze(0).unsqueeze(0).to(device)
        avg_tensor = torch.nn.functional.conv2d(data_tensor, kernel, padding=window_size//2)
        return avg_tensor.squeeze().clamp(0, 255).to('cpu').byte().numpy()
    else:
        avg_map = uniform_filter(data_2d.astype(np.float32), size=window_size, mode='constant')
        return avg_map.astype(np.uint8)

def window_entropy(data_2d: list, i: int, j: int, window_size: int = 3) -> float:
    half = window_size // 2
    values = []
    for x in range(i - half, i + half + 1):
        if x < 0 or x >= len(data_2d):
            continue
        for y in range(j - half, j + half + 1):
            if y < 0 or y >= len(data_2d[x]):
                continue
            values.append(data_2d[x][y])
    if not values:
        return 0.0
    counts = np.bincount(values, minlength=256)
    probs = counts / np.sum(counts)
    return entropy(probs, base=2)

def compute_entropy_map(data_2d: list, window_size: int = 3, step: int = 1, verbose: bool = False, use_gpu: bool = False) -> list:
    if torch.cuda.is_available() and use_gpu:
        data_np = np.array(data_2d, dtype=np.uint8)
        return compute_entropy_map_gpu(data_np, window_size, step).tolist()
    height = len(data_2d)
    width = len(data_2d[0])
    entropy_map = []
    for i in tqdm(range(0, height - window_size + 1, step), desc="Computing entropy map with step", disable=not verbose):
        row = []
        for j in range(0, width - window_size + 1, step):
            row.append(window_entropy(data_2d, i, j, window_size))
        entropy_map.append(row)
    return entropy_map

def compute_entropy_map_gpu(data_2d: np.ndarray, window_size: int = 3, step: int = 1) -> np.ndarray:
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h, w = data_2d.shape
    data_tensor = torch.from_numpy(data_2d).to(device).float().unsqueeze(0).unsqueeze(0)
    patches = data_tensor.unfold(2, window_size, step).unfold(3, window_size, step)
    nh, nw = patches.shape[2], patches.shape[3]
    patches = patches.contiguous().view(1, 1, nh, nw, -1)
    patches = patches.squeeze(0).squeeze(0)
    K = 256
    bins = torch.arange(0, K, device=device).view(1, 1, K)
    patches_exp = patches.unsqueeze(-1)
    hist = (patches_exp == bins).sum(dim=2).float()
    total = hist.sum(dim=-1, keepdim=True) + 1e-8
    probs = hist / total
    entropy_vals = - (probs * torch.log2(probs + 1e-8)).sum(dim=-1)
    return entropy_vals.clamp(0, 8).cpu().numpy()

def window_string_ratio(data_2d: list, i: int, j: int, window_size: int = 3) -> float:
    half = window_size // 2
    total = 0
    string_count = 0
    for x in range(i - half, i + half + 1):
        if x < 0 or x >= len(data_2d):
            continue
        for y in range(j - half, j + half + 1):
            if y < 0 or y >= len(data_2d[x]):
                continue
            total += 1
            if 32 <= data_2d[x][y] <= 126:
                string_count += 1
    return string_count / total if total > 0 else 0.0

def compute_string_ratio_map(data_2d: list, window_size: int = 3, step: int = 1, verbose: bool = False) -> list:
    height = len(data_2d)
    width = len(data_2d[0])
    ratio_map = []
    for i in tqdm(range(0, height - window_size + 1, step), desc="Computing string ratio map with step", disable=not verbose):
        row = []
        for j in range(0, width - window_size + 1, step):
            row.append(window_string_ratio(data_2d, i, j, window_size))
        ratio_map.append(row)
    return ratio_map

def compute_string_ratio_map_np(data_2d: np.ndarray, window_size: int = 3) -> np.ndarray:
    if torch.cuda.is_available():
        device = 'cuda'
        data_tensor = torch.from_numpy(data_2d).to(device)
        printable = ((data_tensor >= 32) & (data_tensor <= 126)).float().unsqueeze(0).unsqueeze(0)
        kernel = torch.ones((1, 1, window_size, window_size), device=device) / (window_size * window_size)
        ratio_tensor = torch.nn.functional.conv2d(printable, kernel, padding=window_size//2)
        return ratio_tensor.squeeze().clamp(0, 1).to('cpu').numpy()
    else:
        printable = ((data_2d >= 32) & (data_2d <= 126)).astype(np.float32)
        ratio_map = uniform_filter(printable, size=window_size, mode='constant')
        return ratio_map

def reshape_as_image(data: np.ndarray, image_height: int, image_width: int) -> np.ndarray:
    if data.ndim == 1:
        return data.reshape((image_height, image_width))
    elif data.ndim == 2 and data.shape[1] == 3:
        return data.reshape((image_height, image_width, 3))
    else:
        raise ValueError(f"Unsupported array shape: {data.shape}")

def compute_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    freq = np.bincount(arr, minlength=256)
    prob = freq / np.sum(freq)
    ent = 0.0
    for p in prob:
        if p > 0:
            ent -= p * math.log2(p)
    return ent
