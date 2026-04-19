import torch
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 65  # Tamaño del vocabulario de TinyShakespeare
    block_size: int = 128 # Longitud de contexto
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    learning_rate: float = 3e-4
    dtype: torch.dtype = torch.bfloat16 # Óptimo para L4 (Ada Lovelace)
    batch_size: int = 256 # Calculado para caber bien en los 24GB de VRAM
    # Auto-detecta si CUDA está disponible
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

config = GPTConfig()
