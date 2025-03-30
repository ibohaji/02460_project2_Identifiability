from dataclasses import dataclass 
from typing import Optional 
import torch 


@dataclass
class TrainingConfig: 
    num_train_data: int = 2048 
    num_classes: int = 3 
    batch_size: int = 32 
    latent_dim: int = 2 
    epochs_per_decoder: int = 100
    base_seed: int = 1000 
    num_vaes: int = 1 
    max_decoder_num: int = 3
    learning_rate: float = 1e-3 
    device: str = "cuda" if torch.cuda.is_available() else "cpu" 


@dataclass 
class GeodesicConfig: 
    num_time_steps: int = 16 
    num_samples: int = 1 
    optimization_steps: int = 500 
    learning_rate: float = 1e-2 
    early_stopping_patience: int = 100 
    early_stopping_delta: float = 1e-4 
    device: str = "cuda" if torch.cuda.is_available() else "cpu" 
