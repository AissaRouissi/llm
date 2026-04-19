"""
LLaMA-style Transformer desde cero.
Implementa: RoPE, RMSNorm, SwiGLU, Grouped-Query Attention (GQA) y KV-Cache.

POR QUÉ cada decisión:
- RoPE: Codifica posición dentro de la atención misma, no como un vector sumado.
  Resultado: el modelo generaliza a secuencias más largas que las vistas en training.
- RMSNorm: Solo normaliza por la raíz cuadrada de la media de cuadrados (sin centrar).
  Resultado: ~15% más rápido que LayerNorm porque elimina la media aritmética.
- SwiGLU: Usa una "compuerta" multiplicativa (gate * up) en lugar de un simple Linear->GELU.
  Resultado: mejor flujo de gradientes y representaciones internas más ricas (paper PaLM).
- GQA: Varias cabezas Q comparten las mismas cabezas K/V.
  Resultado: KV-Cache 2-4x más pequeño durante la inferencia = más rápido y menos VRAM.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .llama_config import LLaMAConfig


# ==============================================================================
# 1. RMSNorm — Reemplazo de LayerNorm (más rápido, sin centrado)
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rsqrt = 1 / sqrt(mean(x^2) + eps)
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ==============================================================================
# 2. RoPE — Rotary Positional Embeddings (sin parámetros aprendidos)
# ==============================================================================
def precompute_rope_cache(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precomputa las frecuencias de rotación para RoPE.
    Cada par de dimensiones del head_dim rota a una frecuencia distinta.
    Las posiciones cercanas tienen embeddings similares; las lejanas, distintos.
    """
    # Frecuencias decrecientes exponencialmente: theta^(-2i/d) para i=0..d/2
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # Posiciones: 0, 1, 2, ..., max_seq_len-1
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # Producto externo: cada posición x cada frecuencia
    freqs = torch.outer(t, freqs)  # (max_seq_len, head_dim/2)
    return freqs.cos(), freqs.sin()  # Cada uno: (max_seq_len, head_dim/2)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0):
    """
    Aplica RoPE a Q o K. Rota pares de dimensiones por ángulos proporcionales a la posición.
    x: (B, n_heads, T, head_dim)
    cos, sin: (max_seq_len, head_dim/2) -> cortamos a [start_pos : start_pos+T]
    """
    seq_len = x.shape[2]
    cos_slice = cos[start_pos:start_pos + seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,T,hd/2)
    sin_slice = sin[start_pos:start_pos + seq_len].unsqueeze(0).unsqueeze(0)

    # Dividir en dos mitades y rotar
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    rotated = torch.cat([x1 * cos_slice - x2 * sin_slice,
                         x2 * cos_slice + x1 * sin_slice], dim=-1)
    return rotated


# ==============================================================================
# 3. Repetición de KV Heads para GQA
# ==============================================================================
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expande las KV heads para alinearlas con el número de Q heads.
    Si n_kv_heads=4 y n_heads=8, cada KV head se repite 2 veces.
    x: (B, n_kv_heads, T, head_dim) -> (B, n_heads, T, head_dim)
    """
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    return (x[:, :, None, :, :]
            .expand(B, n_kv, n_rep, T, hd)
            .reshape(B, n_kv * n_rep, T, hd))


# ==============================================================================
# 4. Grouped-Query Attention (GQA) con soporte para KV-Cache
# ==============================================================================
class GroupedQueryAttention(nn.Module):
    """
    GQA: Múltiples cabezas Q comparten cabezas K/V.
    - n_heads cabezas de Query (completas)
    - n_kv_heads cabezas de Key/Value (compartidas)
    - n_rep = n_heads // n_kv_heads (factor de repetición)
    
    En inferencia, el KV-Cache almacena K,V de tokens previos para no recalcularlos.
    """
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim

        # Proyecciones: Q tiene más cabezas que K/V
        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                start_pos: int = 0, kv_cache: Optional[dict] = None) -> torch.Tensor:
        B, T, _ = x.shape

        # 1. Proyecciones lineales
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, n_heads, T, hd), k/v: (B, n_kv_heads, T, hd)

        # 2. Aplicar RoPE a Q y K (no a V — la posición solo afecta la "relevancia", no el "contenido")
        q = apply_rotary_emb(q, cos, sin, start_pos)
        k = apply_rotary_emb(k, cos, sin, start_pos)

        # 3. KV-Cache (solo en inferencia)
        if kv_cache is not None:
            if 'k' in kv_cache:
                k = torch.cat([kv_cache['k'], k], dim=2)
                v = torch.cat([kv_cache['v'], v], dim=2)
            kv_cache['k'] = k
            kv_cache['v'] = v

        # 4. Expandir KV para GQA (repetir cada KV head n_rep veces)
        k_expanded = repeat_kv(k, self.n_rep)
        v_expanded = repeat_kv(v, self.n_rep)

        # 5. Atención con FlashAttention (SDPA nativo de PyTorch)
        y = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=(kv_cache is None),  # Causal solo en training (en inference el cache ya es causal)
        )

        # 6. Concatenar cabezas y proyectar
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_dropout(self.wo(y))


# ==============================================================================
# 5. SwiGLU MLP — Reemplazo del MLP estándar
# ==============================================================================
class SwiGLUMLP(nn.Module):
    """
    SwiGLU = SiLU(gate) * up, luego down.
    3 matrices en lugar de 2, pero ~misma cantidad de parámetros (hidden_dim es 2/3 de lo normal).
    Resultado empírico: converge más rápido y genera texto más coherente (paper PaLM/LLaMA).
    """
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.w_gate = nn.Linear(config.dim, hidden, bias=False)  # Compuerta
        self.w_up   = nn.Linear(config.dim, hidden, bias=False)  # Expansión
        self.w_down = nn.Linear(hidden, config.dim, bias=False)  # Proyección de vuelta
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate(x)) * up(x), luego down()
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ==============================================================================
# 6. Bloque Transformer (Pre-RMSNorm + GQA + SwiGLU)
# ==============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.feed_forward = SwiGLUMLP(config)

    def forward(self, x, cos, sin, start_pos=0, kv_cache=None):
        # Pre-Norm + Atención + Residual
        h = x + self.attention(self.attention_norm(x), cos, sin, start_pos, kv_cache)
        # Pre-Norm + FFN + Residual
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# ==============================================================================
# 7. Modelo LLaMA Completo
# ==============================================================================
class LLaMA(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config

        # Embedding de tokens (sin embedding posicional — RoPE lo maneja)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)

        # Stack de bloques Transformer
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Normalización final
        self.norm = RMSNorm(config.dim, config.norm_eps)

        # Head de salida (atado al embedding para ahorrar parámetros)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_embeddings.weight = self.lm_head.weight  # Weight tying

        # Precomputar RoPE cache (se registra como buffer, no como parámetro)
        cos, sin = precompute_rope_cache(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

        # Inicialización de pesos
        self.apply(self._init_weights)
        # Escalar la proyección de salida de atención por 1/sqrt(n_layers) (GPT-2 trick)
        for pn, p in self.named_parameters():
            if pn.endswith('wo.weight') or pn.endswith('w_down.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"Secuencia ({T}) > max_seq_len ({self.config.max_seq_len})"

        h = self.dropout(self.tok_embeddings(idx))

        for layer in self.layers:
            if use_gradient_checkpointing and self.training:
                # Gradient Checkpointing: libera activaciones intermedias de la VRAM
                # y las recalcula durante el backward. Ahorra ~60% VRAM a costa de ~30% más tiempo.
                h = grad_checkpoint(layer, h, self.rope_cos, self.rope_sin, 0, None,
                                    use_reentrant=False)
            else:
                h = layer(h, self.rope_cos, self.rope_sin)

        h = self.norm(h)

        if targets is not None:
            logits = self.lm_head(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            logits = self.lm_head(h[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """AdamW con weight decay selectivo (no en normas ni biases)."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay = [p for n, p in param_dict.items() if p.dim() < 2]
        groups = [{'params': decay, 'weight_decay': weight_decay},
                  {'params': nodecay, 'weight_decay': 0.0}]
        use_fused = device_type == 'cuda'
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), fused=use_fused)

    def get_num_params(self):
        """Parámetros únicos (sin contar weight tying dos veces)."""
        return sum(p.numel() for p in self.parameters()) - self.tok_embeddings.weight.numel()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Generación con KV-Cache para máxima velocidad."""
        kv_caches = [dict() for _ in self.layers]

        for token_idx in range(max_new_tokens):
            # En la primera iteración procesamos todo el prompt; después, solo el último token
            if token_idx == 0:
                idx_input = idx
                start_pos = 0
            else:
                idx_input = idx[:, -1:]
                start_pos = idx.shape[1] - 1

            h = self.tok_embeddings(idx_input)
            for i, layer in enumerate(self.layers):
                h = layer(h, self.rope_cos, self.rope_sin, start_pos, kv_caches[i])
            h = self.norm(h)
            logits = self.lm_head(h[:, -1, :])

            # Sampling
            if temperature == 0.0:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[mask] = -float('inf')
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx
