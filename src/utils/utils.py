import torch
import numpy as np
from typing import Tuple, Optional
import random
import os


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the device to use for training.
    
    Args:
        device: Optional device string ('cuda' or 'cpu')
        
    Returns:
        torch.device object
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def save_image_grid(
    images: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Tuple[float, float] = (-1, 1),
):
    """
    Save a grid of images.
    
    Args:
        images: Tensor of images (B, C, H, W)
        save_path: Path to save the grid
        nrow: Number of images per row
        normalize: Whether to normalize the images
        value_range: Range of values for normalization
    """
    from torchvision.utils import save_image
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(
        images,
        save_path,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range
    )


def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """
    Compute Fréchet Inception Distance between real and fake images.
    
    Args:
        real_images: Tensor of real images (B, C, H, W)
        fake_images: Tensor of fake images (B, C, H, W)
        
    Returns:
        FID score
    """
    # TODO: Implement FID computation
    # This will require:
    # - Loading a pre-trained Inception model
    # - Computing features for both real and fake images
    # - Computing the FID score
    raise NotImplementedError("FID computation not implemented yet")


def compute_psnr(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between real and fake images.
    
    Args:
        real_images: Tensor of real images (B, C, H, W)
        fake_images: Tensor of fake images (B, C, H, W)
        
    Returns:
        PSNR score
    """
    mse = torch.mean((real_images - fake_images) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item() 