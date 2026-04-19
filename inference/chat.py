import os
import sys
import torch
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.generate import load_model_from_checkpoint, sample
from models.gpt_config import GPTConfig
from models.tokenizer import CharTokenizer

# Colores ANSI
BLUE = "\033[94m"
GREEN = "\033[92m"
RESET = "\033[0m"

def supports_color():
    """Detecta si el terminal soporta colores ANSI."""
    plat = sys.platform
    supported_platform = plat != 'Pocket PC' and (plat != 'win32' or 'ANSICON' in os.environ)
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    return supported_platform and is_a_tty

if not supports_color():
    BLUE = GREEN = RESET = ""

def format_prompt(instruction):
    return f"### Instrucción:\n{instruction}\n\n### Respuesta:\n"

@torch.no_grad()
def chat_generate(model, tokenizer, prompt, max_tokens, device, temperature=0.7, top_p=0.9):
    """Generación especial para chat que se detiene al detectar inicio de nueva instrucción."""
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    response_chars = []
    buffer = ""
    stop_sequence = "### Instrucción:" # Condición de parada temprana
    
    print(GREEN, end='', flush=True)
    
    for _ in range(max_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        
        idx_next = sample(logits, temperature=temperature, top_p=top_p)
        idx = torch.cat((idx, idx_next), dim=1)
        
        next_char = tokenizer.decode(idx_next[0].tolist())
        buffer += next_char
        response_chars.append(next_char)
        
        # Parada temprana
        if stop_sequence in buffer:
            # Eliminar la secuencia de parada de lo que hemos impreso/guardado
            response_chars = response_chars[:-len(stop_sequence)]
            break
            
        print(next_char, end='', flush=True)
        
        # Mantener el buffer pequeño por eficiencia
        if len(buffer) > len(stop_sequence) * 2:
            buffer = buffer[-len(stop_sequence):]
            
    print(RESET)
    return "".join(response_chars).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/finetune/finetuned_best.pt')
    parser.add_argument('--vocab', type=str, default='data/processed/vocab.json')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--max-tokens', type=int, default=500)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = CharTokenizer()
    if os.path.exists(args.vocab):
        tokenizer.load(args.vocab)
    else:
        print("ERROR: No se encontró el vocabulario.")
        return
        
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: No se encontró el modelo finetuneado en {args.checkpoint}")
        print("Usando modelo con pesos aleatorios para testing.")
        model = GPT(GPTConfig())
        model.to(device)
        model.eval()
    else:
        model, _ = load_model_from_checkpoint(args.checkpoint, device)
        
    print("\n" + "="*50)
    print("🤖 CHATBOT MINI-LLM INICIADO")
    print("Comandos: /reset (limpia historial), /save (guarda chat), /exit")
    print("="*50 + "\n")
    
    history = []
    
    try:
        while True:
            user_input = input(f"{BLUE}Usuario:{RESET} ")
            if not user_input.strip():
                continue
                
            if user_input == "/exit":
                break
            elif user_input == "/reset":
                history = []
                print("--- Historial limpiado ---")
                continue
            elif user_input == "/save":
                filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    for role, text in history:
                        f.write(f"{role}: {text}\n\n")
                print(f"--- Chat guardado en {filename} ---")
                continue
                
            # Formatear el prompt como en el fine-tuning
            prompt = format_prompt(user_input)
            
            # Generar respuesta
            response = chat_generate(model, tokenizer, prompt, args.max_tokens, device, 
                                     temperature=args.temperature, top_p=args.top_p)
            
            # Guardar en historial
            history.append(("Usuario", user_input))
            history.append(("Modelo", response))
            
    except KeyboardInterrupt:
        print("\n¡Adiós!")

if __name__ == "__main__":
    main()
