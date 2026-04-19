import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.generate import load_model_from_checkpoint, generate
from models.gpt_config import GPTConfig
from models.gpt_model import GPT
from models.tokenizer import CharTokenizer

def compare_strategies():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = CharTokenizer()
    vocab_path = "data/processed/vocab.json"
    if os.path.exists(vocab_path):
        tokenizer.load(vocab_path)
    else:
        tokenizer.fit("abcdefghijklmnopqrstuvwxyz \n")
        
    ckpt_path = "checkpoints/ckpt_best.pt"
    if os.path.exists(ckpt_path):
        model, _ = load_model_from_checkpoint(ckpt_path, device)
    else:
        print("[AVISO] No hay modelo entrenado, usando pesos aleatorios en la comparación.")
        model = GPT(GPTConfig(device=device))
        model.to(device)
        model.eval()
        
    prompt = "El secreto del universo es"
    max_tok = 80
    
    print("\n" + "="*50)
    print("🧠 COMPARACIÓN DE ESTRATEGIAS DE SAMPLING")
    print("="*50)
    
    # 1. Greedy
    print("\n[1] GREEDY (temp=0.0)")
    print("Determinista, siempre escoge el token más probable. Suele crear bucles infinitos en LMs pequeños.")
    generate(model, tokenizer, prompt, max_tok, device, temperature=0.0, top_k=None, top_p=None)
    
    # 2. Conservative
    print("\n\n[2] CONSERVATIVE (temp=0.5, top_k=20)")
    print("Seguro pero con un toque de variabilidad. Respuestas predecibles.")
    generate(model, tokenizer, prompt, max_tok, device, temperature=0.5, top_k=20, top_p=None)
    
    # 3. Balanced
    print("\n\n[3] BALANCED (temp=0.8, top_k=50)")
    print("El estándar de la industria. Buen balance entre coherencia y creatividad.")
    generate(model, tokenizer, prompt, max_tok, device, temperature=0.8, top_k=50, top_p=None)
    
    # 4. Creative
    print("\n\n[4] CREATIVE (temp=1.2, top_p=0.95)")
    print("Caótico. Usa Nucleus Sampling (top_p) para cortar la 'larga cola' de tokens absurdos e intentar que no rompa el idioma.")
    generate(model, tokenizer, prompt, max_tok, device, temperature=1.2, top_k=None, top_p=0.95)
    
    print("\n\n" + "="*50)

if __name__ == "__main__":
    compare_strategies()
