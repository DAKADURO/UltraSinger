"""GPU memory helpers: release VRAM between pipeline stages and pick safe defaults for small GPUs"""
import gc

# Free VRAM thresholds (GB) used to pick whisper defaults when the user did not choose any
_VRAM_COMFORTABLE_GB = 14
_VRAM_MEDIUM_GB = 9
_VRAM_SMALL_GB = 6


def free_gpu_memory() -> None:
    """Drop unreferenced objects and hand cached CUDA blocks back to the driver"""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_gpu_total_memory_gb(device_index: int = 0) -> float | None:
    """Total VRAM of the CUDA device in GB, None if there is no usable CUDA device"""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(device_index).total_memory / 1024 ** 3


def recommend_whisper_settings(device: str, vram_gb: float | None) -> tuple[int, str]:
    """Return (batch_size, compute_type) that fits the available memory.

    The WhisperX defaults (batch 16, float16) need more than 8 GB with the large models.
    """
    if device != "cuda" or vram_gb is None:
        return 8, "int8"
    if vram_gb >= _VRAM_COMFORTABLE_GB:
        return 16, "float16"
    if vram_gb >= _VRAM_MEDIUM_GB:
        return 8, "float16"
    if vram_gb >= _VRAM_SMALL_GB:
        return 6, "int8_float16"
    return 2, "int8_float16"


def resolve_whisper_settings(device: str, batch_size: int | None, compute_type: str | None,
                             vram_gb: float | None = None) -> tuple[int, str]:
    """Fill batch size / compute type the user left open with values recommended for this GPU"""
    if batch_size is not None and compute_type is not None:
        return batch_size, compute_type
    if vram_gb is None and device == "cuda":
        vram_gb = get_gpu_total_memory_gb()
    rec_batch, rec_compute = recommend_whisper_settings(device, vram_gb)
    return (batch_size if batch_size is not None else rec_batch,
            compute_type if compute_type is not None else rec_compute)
