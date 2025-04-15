import os
import pefile
import numpy as np
import ray
from . import utils
from . import r_channel
from . import g_channel
from . import b_channel
from . import transform

class PE2Image:
    def __init__(self, file_path: str, pe_image_width: int = 256, window_size: int = 32,
                 verbose: bool = False, use_gpu: bool = False) -> None:
        self.file_path = file_path
        try:
            self.pe = pefile.PE(self.file_path)
        except pefile.PEFormatError:
            raise ValueError(f"[ERROR] {self.file_path} is not a valid PE file.")
        self.pe_size_original = len(self.pe.__data__)
        self.pe_image_width = pe_image_width
        self.window_size = window_size
        self.verbose = verbose
        self.use_gpu = use_gpu

        self.pe_padding_size = (self.pe_image_width - (self.pe_size_original % self.pe_image_width)) % self.pe_image_width
        self.pe_rgb_map = np.full((self.pe_size_original + self.pe_padding_size, 3), [0, 0, 0], dtype=np.uint8)

        self.pe_map = np.concatenate((
            np.frombuffer(self.pe.__data__, dtype=np.uint8),
            np.zeros(self.pe_padding_size, dtype=np.uint8)
        ))
        self.pe_size_new = self.pe_map.shape[0]
        self.pe_image_height = self.pe_size_new // self.pe_image_width

    def reshape_as_image(self, data: np.ndarray) -> np.ndarray:
        return utils.reshape_as_image(data, self.pe_image_height, self.pe_image_width)

    def apply_entropy_to_r(self, step: int = 1) -> None:
        r_channel.apply_entropy_to_r(self, step)

    def apply_string_ratio_to_g(self, step: int = 1) -> None:
        g_channel.apply_string_ratio_to_g(self, step)

    def apply_section_permissions_to_b(self) -> None:
        b_channel.apply_section_permissions_to_b(self)

    def transform_to_fixed_size_interpolation(self, output_size: tuple = (256, 256)) -> np.ndarray:
        return transform.transform_to_fixed_size_interpolation(self, output_size)

    def save_fixed_interpolation(self, base_output_dir: str, file_name: str, output_size: tuple = (256, 256)) -> None:
        transform.save_fixed_interpolation(self, base_output_dir, file_name, output_size)

    def save_channel_color_images(self, base_output_dir: str, file_name: str, output_size: tuple = (256, 256)) -> None:
        transform.save_channel_color_images(self, base_output_dir, file_name, output_size)

    def save_original_image(self, base_output_dir: str, file_name: str) -> None:
        transform.save_original_image(self, base_output_dir, file_name)

    def make_image(self, base_output_dir: str, step_size: int = 1) -> str:
        file_base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        self.apply_entropy_to_r(step_size)
        self.apply_string_ratio_to_g(step_size)
        self.apply_section_permissions_to_b()
        # save original image before interpolation
        self.save_original_image(base_output_dir, file_base_name)
        self.save_fixed_interpolation(base_output_dir, file_base_name)
        self.save_channel_color_images(base_output_dir, file_base_name)
        return file_base_name

# Ray Actors Option
PE2Image = ray.remote(PE2Image)
