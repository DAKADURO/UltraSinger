"""Device detection module."""

from typing import Optional

import torch

from modules.console_colors import ULTRASINGER_HEAD, red_highlighted, blue_highlighted

pytorch_gpu_supported = False

def check_gpu_support() -> str:
    """Check worker device (e.g cuda or cpu) supported by pytorch"""

    print(f"{ULTRASINGER_HEAD} Checking GPU support.")

    pytorch_gpu_supported = __check_pytorch_support()

    return 'cuda' if pytorch_gpu_supported else 'cpu'


def get_gpu_vram_gb() -> Optional[float]:
    """Total VRAM of the first cuda device in GB, None if there is no cuda device"""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def recommend_whisper_settings(vram_gb: Optional[float]) -> tuple[int, Optional[str]]:
    """Whisper batch size and compute type that fit into the given VRAM (large-v2 model).
    A compute type of None keeps the default of whisper (float16 on cuda, int8 on cpu)."""
    if vram_gb is None:
        return 16, None
    if vram_gb >= 16:
        return 16, None
    if vram_gb >= 10:
        return 8, None
    if vram_gb >= 7:
        return 8, "int8"
    return 4, "int8"


def __check_pytorch_support():
    pytorch_gpu_supported = torch.cuda.is_available()
    if not pytorch_gpu_supported:
        print(
            f"{ULTRASINGER_HEAD} {blue_highlighted('pytorch')} - there are no {red_highlighted('cuda')} devices available -> Using {red_highlighted('cpu')}."
        )
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_properties = torch.cuda.get_device_properties(0)
        gpu_vram = round(gpu_properties.total_memory / 1024 ** 3, 2)  # Convert bytes to GB and round to 2 decimal places
        print(f"{ULTRASINGER_HEAD} Found GPU: {blue_highlighted(gpu_name)} VRAM: {blue_highlighted(gpu_vram)} GB.")
        if gpu_vram < 6:
            print(
                f"{ULTRASINGER_HEAD} {red_highlighted('GPU VRAM is less than 6GB. Program may crash due to insufficient memory.')}")
        print(f"{ULTRASINGER_HEAD} {blue_highlighted('pytorch')} - using {red_highlighted('cuda')} gpu.")
    return pytorch_gpu_supported
