import os
import sys
import torch

# Asegurar importación desde la raíz
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gpt_config import GPTConfig
from models.gpt_model import GPT

def run_test():
    print("Iniciando test del modelo Transformer en CPU (MacBook)...")
    config = GPTConfig(vocab_size=65, block_size=128, n_layer=2, n_head=2, n_embd=64, device='cpu')
    
    print(f"[DEBUG] Instanciando GPT: n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}")
    model = GPT(config)
    print(f"[DEBUG] Parámetros totales: {model.get_num_parameters():,}")
    
    B, T = 4, 32
    print(f"[DEBUG] Creando dummy tensor inputs con shape B={B}, T={T}")
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    
    model.eval()
    print("Ejecutando forward pass (Inferencia)...")
    logits_inf, loss_inf = model(idx)
    print(f"[DEBUG] Logits shape (Inferencia): {logits_inf.shape} - Esperado: ({B}, 1, {config.vocab_size})")
    assert logits_inf.shape == (B, 1, config.vocab_size)
    assert loss_inf is None
    
    model.train()
    print("Ejecutando forward pass (Train)...")
    logits_train, loss_train = model(idx, targets)
    print(f"[DEBUG] Logits shape (Train): {logits_train.shape} - Esperado: ({B}, {T}, {config.vocab_size})")
    print(f"[DEBUG] Loss: {loss_train.item():.4f}")
    assert logits_train.shape == (B, T, config.vocab_size)
    
    print("Ejecutando backward pass...")
    loss_train.backward()
    print("[DEBUG] Backward pass exitoso. No hay errores dinámicos de grafo.")
    
    model.eval()
    print("Ejecutando generate()...")
    idx_init = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(idx_init, max_new_tokens=10)
    print(f"[DEBUG] Tokens generados (shape): {generated.shape}")
    
    print("\n✅ TEST COMPLETADO SIN ERRORES - ARQUITECTURA SANA")

if __name__ == "__main__":
    run_test()
