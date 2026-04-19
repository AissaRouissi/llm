"""Test rápido del modelo LLaMA-style en CPU."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.llama_config import get_small_config
from models.llama_model import LLaMA

def test_llama():
    print("🧪 Test de arquitectura LLaMA-style...")
    
    config = get_small_config(vocab_size=1000, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, 
                               max_seq_len=32, device='cpu')
    
    model = LLaMA(config)
    n = model.get_num_params()
    print(f"✅ Modelo creado: {n:,} parámetros ({n/1e6:.2f}M)")
    
    # Test forward pass
    x = torch.randint(0, 1000, (2, 16))
    y = torch.randint(0, 1000, (2, 16))
    logits, loss = model(x, y)
    print(f"✅ Forward pass OK. Loss: {loss.item():.4f}")
    
    # Test backward
    loss.backward()
    print("✅ Backward pass OK.")
    
    # Test gradient checkpointing
    model.zero_grad()
    logits, loss = model(x, y, use_gradient_checkpointing=True)
    loss.backward()
    print("✅ Gradient Checkpointing OK.")
    
    # Test generación con KV-Cache
    model.eval()
    prompt = torch.randint(0, 1000, (1, 5))
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=50)
    print(f"✅ Generación con KV-Cache OK. Output shape: {out.shape}")
    
    # Test GQA
    print(f"✅ GQA: {config.n_heads} Q heads, {config.n_kv_heads} KV heads (ratio {config.n_heads//config.n_kv_heads}:1)")
    
    print("\n🏆 TODOS LOS TESTS PASADOS. Arquitectura LLaMA lista.")

if __name__ == '__main__':
    test_llama()
