import os
import sys
import math
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.generate import load_model_from_checkpoint
from models.tokenizer import CharTokenizer
from models.gpt_config import GPTConfig
from models.gpt_model import GPT

@torch.no_grad()
def calculate_perplexity(model, tokenizer, text, device):
    """Calcula la perplejidad del modelo iterando sobre bloques del tamaño del contexto."""
    model.eval()
    tokens = tokenizer.encode(text)
    if len(tokens) <= 1:
        return float('inf')
        
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    block_size = model.config.block_size
    nlls = []
    
    # Procesar en ventanas
    for i in range(0, idx.size(1) - 1, block_size):
        end_idx = min(i + block_size, idx.size(1) - 1)
        if i == end_idx: break
            
        x = idx[:, i:end_idx]
        y = idx[:, i+1:end_idx+1]
        
        logits, loss = model(x, y)
        # La loss es el negative log likelihood medio del batch.
        # Multiplicamos por la longitud real del chunk para luego hacer media ponderada
        nlls.append(loss.item() * (end_idx - i))
        
    total_tokens = idx.size(1) - 1
    avg_loss = sum(nlls) / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = 'checkpoints/ckpt_best.pt'
    vocab_path = 'data/processed/vocab.json'
    
    tokenizer = CharTokenizer()
    if os.path.exists(vocab_path):
        tokenizer.load(vocab_path)
    else:
        print("[ERROR] Vocabulario no encontrado.")
        return
        
    if os.path.exists(ckpt_path):
        model, _ = load_model_from_checkpoint(ckpt_path, device)
    else:
        print("[AVISO] No hay checkpoint. Usando pesos aleatorios.")
        model = GPT(GPTConfig(device=device))
        model.to(device)
        
    text_train = "ROMEO:\nAntes de proseguir, escuchadme."
    text_val = "JULIETA:\nSi profano con mi indigna mano este sagrado relicario."
    text_novel = "La inteligencia artificial avanza rápidamente en la computación moderna."
    
    print("\n" + "="*50)
    print("📊 EVALUACIÓN DE PERPLEJIDAD (PPL)")
    print("="*50)
    
    ppl_train = calculate_perplexity(model, tokenizer, text_train, device)
    print(f"-> PPL en texto similar al Training: {ppl_train:.2f}")
    
    ppl_val = calculate_perplexity(model, tokenizer, text_val, device)
    print(f"-> PPL en texto similar a Validación: {ppl_val:.2f}")
    
    ppl_novel = calculate_perplexity(model, tokenizer, text_novel, device)
    print(f"-> PPL en texto completamente nuevo (Out of Distribution): {ppl_novel:.2f}")
    
    print("\n[🔍 Interpretación]")
    print("PPL < 20 : Excelente comprensión de las reglas del lenguaje.")
    print("PPL 20-50: Entendimiento decente, generará texto legible.")
    print("PPL > 100: Modelo pobre (pesos aleatorios ~ vocab_size). Nivel adivinación.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
