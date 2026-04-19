import os
import urllib.request
import torch
from models.tokenizer import CharTokenizer
from models.gpt_config import config

def download_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = "data/raw/tinyshakespeare.txt"
    
    if not os.path.exists(filepath):
        print(f"Descargando dataset desde {url}...")
        urllib.request.urlretrieve(url, filepath)
        print("Descarga completada.")
    else:
        print("Dataset ya descargado.")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    # 1. Obtener datos
    text = download_data()
    print(f"Longitud del texto: {len(text):,} caracteres")
    
    # 2. Entrenar y guardar Tokenizador
    tok = CharTokenizer()
    tok.fit(text)
    vocab_path = "data/processed/vocab.json"
    tok.save(vocab_path)
    print(f"Vocabulario de tamaño {len(tok)} guardado en {vocab_path}")
    
    # 3. Convertir texto a tensores
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    torch.save(data, "data/processed/data.pt")
    
    # 4. Cálculos para dataloader/entrenamiento
    batches_per_epoch = len(data) // (config.batch_size * config.block_size)
    
    print("\n--- RESUMEN FINAL ---")
    print(f"Vocab size: {len(tok)}")
    print(f"Dataset size: {len(data):,} tokens")
    print(f"Batches por epoch (estimado): {batches_per_epoch}")
    
    # 5. Test rápido de dataloader
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    
    print(f"\nTest rápido de batching (batch_size={config.batch_size}, block_size={config.block_size}):")
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")

if __name__ == "__main__":
    main()
