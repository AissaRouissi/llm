import os
import sys
import torch
import torch.nn.functional as F
import argparse

# Asegurar importaciones desde el directorio raíz
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gpt_config import GPTConfig
from models.gpt_model import GPT
from models.tokenizer import CharTokenizer

def load_model_from_checkpoint(checkpoint_path, device):
    """Carga config, modelo y pesos desde un .pt/.ckpt y lo pone en modo inferencia."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} no encontrado.")
        
    print(f"[INFO] Cargando checkpoint desde {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get('config', GPTConfig())
    model = GPT(config)
    
    # Remover el prefijo '_orig_mod.' si el modelo fue guardado con torch.compile()
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"[INFO] Modelo listo. Parámetros entrenables: {model.get_num_parameters():,}")
    if 'val_loss' in checkpoint:
        print(f"[INFO] Val Loss en checkpoint: {checkpoint['val_loss']:.4f}")
        
    return model, config

def sample(logits, temperature=1.0, top_k=None, top_p=None):
    """
    Estrategias de sampling: Temperature, Top-K y Nucleus (Top-P).
    Las operaciones se hacen antes del softmax para mayor estabilidad numérica.
    """
    # Tomar solo la predicción del último token
    logits = logits[:, -1, :] # Shape: (B, vocab_size)
    
    # Greedy decoding directo si temperature == 0
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
        
    # Aplicar temperatura
    logits = logits / temperature
    
    # Top-K filtering: mantenemos solo los k tokens con mayor probabilidad
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        
    # Top-P (Nucleus) filtering: cortamos la cola de la distribución acumulada
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
        
    # Extraer de la distribución de probabilidad resultante
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    return idx_next

@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens, device, temperature=0.8, top_k=50, top_p=0.9):
    """
    Inferencia token a token con output en streaming por consola.
    """
    print(prompt, end='', flush=True)
    
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    # TODO: Implementar KV-cache en gpt_model.py para inferencia en O(1) en vez de O(N^2)
    
    for _ in range(max_tokens):
        # Truncar secuencia al contexto máximo que soporta el modelo
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        
        # Forward pass rápido
        logits, _ = model(idx_cond)
        
        # Samplear según estrategias
        idx_next = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        
        # Agregar al historial
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Stream directo a la consola
        next_char = tokenizer.decode(idx_next[0].tolist())
        print(next_char, end='', flush=True)
        
    print() # Salto de línea final
    return idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/ckpt_best.pt', help='Path to checkpoint')
    parser.add_argument('--vocab', type=str, default='data/processed/vocab.json', help='Path to vocab.json')
    parser.add_argument('--prompt', type=str, default='', help='Texto inicial')
    parser.add_argument('--max-tokens', type=int, default=200, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='>1 is random, 0 is deterministic')
    parser.add_argument('--top-k', type=int, default=50, help='Keep only top K tokens')
    parser.add_argument('--top-p', type=float, default=0.9, help='Keep tokens accumulating P prob')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Setup del tokenizador
    tokenizer = CharTokenizer()
    if os.path.exists(args.vocab):
        tokenizer.load(args.vocab)
    else:
        print("[AVISO] No se encontró vocab.json. El output será aleatorio y sin sentido.")
        tokenizer.fit("abcdefghijklmnopqrstuvwxyz \n")
        
    # Setup del modelo
    if not os.path.exists(args.checkpoint):
        print(f"[AVISO] Checkpoint '{args.checkpoint}' no existe. Usando pesos aleatorios.")
        config = GPTConfig()
        model = GPT(config)
        model.to(args.device)
        model.eval()
    else:
        model, _ = load_model_from_checkpoint(args.checkpoint, args.device)
        
    # Modo de un solo tiro
    if args.prompt:
        generate(model, tokenizer, args.prompt, args.max_tokens, args.device, 
                 args.temperature, args.top_k, args.top_p)
    # Modo Interactivo
    else:
        print("\n=== CHAT INTERACTIVO (Presiona Ctrl+C para salir) ===")
        try:
            while True:
                user_prompt = input("\nPrompt > ")
                if not user_prompt:
                    continue
                generate(model, tokenizer, user_prompt, args.max_tokens, args.device, 
                         args.temperature, args.top_k, args.top_p)
        except KeyboardInterrupt:
            print("\n¡Desconectado!")

if __name__ == '__main__':
    main()
