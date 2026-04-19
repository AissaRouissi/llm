"""
Configuración del modelo LLaMA-style.
Dos presets: 'small' (Kaggle P100/T4 16GB) y 'medium' (L4 24GB).
"""
from dataclasses import dataclass

@dataclass
class LLaMAConfig:
    # --- Tokenizador ---
    vocab_size: int = 32000       # SentencePiece o tiktoken (ajustable)

    # --- Dimensiones del Modelo ---
    dim: int = 512                # Dimensión de embedding
    n_layers: int = 8             # Bloques Transformer
    n_heads: int = 8              # Cabezas de Query
    n_kv_heads: int = 4           # Cabezas de Key/Value (GQA: menos KV que Q)
    max_seq_len: int = 512        # Contexto máximo
    
    # --- SwiGLU MLP ---
    # hidden_dim se calcula automáticamente: 2/3 * 4 * dim, redondeado
    multiple_of: int = 256        # Alineación de hidden_dim (eficiencia GPU)
    
    # --- Regularización ---
    dropout: float = 0.0          # LLaMA no usa dropout (modelo grande + datos masivos)
    
    # --- RoPE ---
    rope_theta: float = 10000.0   # Frecuencia base para Rotary Embeddings
    
    # --- Normalización ---
    norm_eps: float = 1e-5        # Epsilon para RMSNorm
    
    # --- Entrenamiento ---
    learning_rate: float = 3e-4
    batch_size: int = 32
    
    # --- Device ---
    device: str = 'cuda'

    @property
    def head_dim(self):
        return self.dim // self.n_heads

    @property
    def hidden_dim(self):
        """Calcula la dimensión oculta del MLP SwiGLU estilo LLaMA."""
        h = int(2 * (4 * self.dim) / 3)
        return self.multiple_of * ((h + self.multiple_of - 1) // self.multiple_of)

# --- PRESETS ---
def get_small_config(**overrides):
    """~50M params. Para Kaggle P100/T4 (16GB VRAM)."""
    cfg = LLaMAConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=4, max_seq_len=512)
    for k, v in overrides.items(): setattr(cfg, k, v)
    return cfg

def get_medium_config(**overrides):
    """~150M params. Para NVIDIA L4 (24GB VRAM)."""
    cfg = LLaMAConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=1024, batch_size=16)
    for k, v in overrides.items(): setattr(cfg, k, v)
    return cfg
