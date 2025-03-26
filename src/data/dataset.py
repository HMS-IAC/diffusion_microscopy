import os
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class MicroscopyDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (256, 256),
    ):
        """
        Dataset class for microscopy images.
        
        Args:
            root_dir: Directory containing the microscopy images
            transform: Optional albumentations transform pipeline
            image_size: Target size for the images (height, width)
        """
        self.root_dir = root_dir
        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ]
        
        if transform is None:
            self.transform = A.Compose([
                A.Resize(*image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a microscopy image.
        
        Args:
            idx: Index of the image to load
            
        Returns:
            Tensor containing the preprocessed image
        """
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            transformed = self.transform(image=np.array(image))
            image = transformed['image']
            
        return image 