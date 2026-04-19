import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gpt_config import GPTConfig
from models.gpt_model import GPT
from inference.generate import sample

def run_test():
    print("Iniciando test de generación (CPU)...")
    
    config = GPTConfig(vocab_size=10, block_size=16, n_layer=1, n_head=1, n_embd=16, device='cpu')
    model = GPT(config)
    model.eval()
    
    # Batch=1, Sequence=1, Vocab=10
    logits = torch.randn(1, 1, 10)
    logits[0, 0, 5] = 100.0 # Hacemos que el token 5 sea casi seguro
    
    # 1. Test Greedy (temp=0)
    token_greedy = sample(logits, temperature=0.0)
    assert token_greedy.item() == 5
    print("✅ Greedy sampling (temp=0) funciona y es determinista.")
    
    # 2. Test Top-K
    logits_flat = torch.zeros(1, 1, 10)
    logits_flat[0, 0, 2] = 10.0
    logits_flat[0, 0, 7] = 8.0
    # Con top_k=2 solo podemos obtener 2 o 7
    token_topk = sample(logits_flat, temperature=1.0, top_k=2)
    assert token_topk.item() in [2, 7]
    print("✅ Top-K filtering recorta correctamente los peores tokens.")
    
    # 3. Test Top-P (Nucleus)
    logits_p = torch.tensor([[[10.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    # 10.0 y 9.0 acumulan más de 0.999 tras softmax
    token_topp = sample(logits_p, temperature=1.0, top_p=0.9)
    assert token_topp.item() in [0, 1]
    print("✅ Nucleus sampling (Top-P) filtra la larga cola con éxito.")
    
    # 4. Verificar wrapper del modelo
    idx = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=5, temperature=0.8)
    assert out.shape == (1, 8)
    print("✅ model.generate() enlaza el bucle completo y produce n tokens.")
    
    print("\n✅ TEST CPU DE INFERENCIA COMPLETADO SIN ERRORES")

if __name__ == '__main__':
    run_test()
