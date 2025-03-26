from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    image_size: Tuple[int, int] = (256, 256)
    model_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int] = (16,)
    dropout: float = 0.1
    channel_mult: Tuple[int] = (1, 2, 3, 4)
    num_workers: int = 4
    pin_memory: bool = True
    checkpoint_frequency: int = 5


@dataclass
class DiffusionConfig:
    """Configuration for diffusion process."""
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = 'linear'
    prediction_type: str = 'epsilon'  # 'epsilon' or 'x0'


@dataclass
class DataConfig:
    """Configuration for data processing."""
    train_dir: str = 'data/raw'
    processed_dir: str = 'data/processed'
    image_size: Tuple[int, int] = (256, 256)
    num_channels: int = 3
    augment: bool = True
    normalize: bool = True


@dataclass
class Config:
    """Main configuration class."""
    training: TrainingConfig = TrainingConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    data: DataConfig = DataConfig()
    log_dir: str = 'logs'
    save_dir: str = 'checkpoints'
    device: Optional[str] = None  # Will be set to 'cuda' if available
    seed: int = 42 