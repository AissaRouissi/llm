import os
import sys
import json
import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tokenizer import CharTokenizer

def format_example(instruction, output):
    """Template básico tipo Alpaca para instrucción y respuesta."""
    text = f"### Instrucción:\n{instruction}\n\n### Respuesta:\n{output}\n"
    prompt = f"### Instrucción:\n{instruction}\n\n### Respuesta:\n"
    return text, prompt

def create_sample_dataset(filepath):
    """Genera 50 ejemplos falsos si no se provee un dataset."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    examples = [
        {"instruction": "¿Qué es la IA?", "output": "Inteligencia Artificial."},
        {"instruction": "Di hola.", "output": "Hola."}
    ] * 25
    with open(filepath, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f"[INFO] Dataset ficticio creado en {filepath}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/sample_dataset.jsonl')
    parser.add_argument('--vocab', type=str, default='data/processed/vocab.json')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        create_sample_dataset(args.input)
        
    tokenizer = CharTokenizer()
    if os.path.exists(args.vocab):
        tokenizer.load(args.vocab)
    else:
        print("[ERROR] Vocabulario no encontrado. Fase 1 requerida.")
        return
        
    data_x, data_y = [], []
    
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            text, prompt = format_example(ex['instruction'], ex['output'])
            
            text_ids = tokenizer.encode(text)
            prompt_ids = tokenizer.encode(prompt)
            
            # Instruction Masking: El modelo solo debe aprender a generar la "Respuesta"
            y = text_ids.copy()
            y[:len(prompt_ids)] = -100 # -100 hace que PyTorch ignore estos tokens en la Loss
            
            data_x.extend(text_ids)
            data_y.extend(y)
            
    # Autoregressive shift: X[:-1] predice Y[1:]
    # Gracias a esto, el último token del prompt predice el primer token de la respuesta.
    # El salto entre documentos tendrá Y=-100, evitando que memorice secuencias entre ellos.
    X_tensor = torch.tensor(data_x[:-1], dtype=torch.long)
    Y_tensor = torch.tensor(data_y[1:], dtype=torch.long)
    
    n_val = max(1, int(len(X_tensor) * 0.1))
    n_train = len(X_tensor) - n_val
    
    train_data = {'x': X_tensor[:n_train], 'y': Y_tensor[:n_train]}
    val_data = {'x': X_tensor[n_train:], 'y': Y_tensor[n_train:]}
    
    os.makedirs("data/processed", exist_ok=True)
    torch.save(train_data, "data/processed/finetune_train.pt")
    torch.save(val_data, "data/processed/finetune_val.pt")
    
    print(f"[INFO] Fine-tuning dataset guardado en data/processed/")
    print(f"       Train tokens: {len(train_data['x']):,} | Val tokens: {len(val_data['x']):,}")

if __name__ == "__main__":
    main()
