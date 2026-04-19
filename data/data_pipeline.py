"""
Pipeline de datos profesional para pre-entrenamiento de LLMs.

Flujo:
1. Recopila texto de data/raw/ (cualquier .txt que haya)
2. Tokeniza con tiktoken (cl100k_base, el tokenizador de GPT-4)
3. Guarda como numpy memmap (lectura directa desde disco, sin cargar en RAM)
4. Crea un DataLoader eficiente para el training loop

POR QUÉ tiktoken en lugar de CharTokenizer:
- CharTokenizer: 1 carácter = 1 token. "inteligencia" = 12 tokens.
- tiktoken BPE: palabras frecuentes son 1 token. "inteligencia" = 1-2 tokens.
- Resultado: el mismo texto ocupa 4x menos tokens → el modelo "ve" 4x más contexto.

POR QUÉ memmap en lugar de torch.load:
- torch.load carga TODO el dataset en RAM. Con 2GB de texto, necesitas 2GB+ de RAM.
- memmap lee los tokens directamente del disco, bloque a bloque. Usa ~0 RAM extra.
"""
import os
import glob
import argparse
import numpy as np

try:
    import tiktoken
except ImportError:
    print("[AVISO] tiktoken no instalado. Ejecuta: pip install tiktoken")
    tiktoken = None


def collect_raw_text(raw_dir: str) -> str:
    """Lee todos los .txt de la carpeta raw y los concatena."""
    files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if not files:
        print(f"[ERROR] No hay archivos .txt en {raw_dir}")
        return ""

    all_text = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                text = fh.read()
                all_text.append(text)
                print(f"  -> {os.path.basename(f)}: {len(text):,} caracteres")
        except Exception as e:
            print(f"  [SKIP] {f}: {e}")

    combined = "\n\n".join(all_text)
    print(f"\n[INFO] Texto total recopilado: {len(combined):,} caracteres")
    return combined


def tokenize_text(text: str, encoding_name: str = "cl100k_base") -> np.ndarray:
    """Tokeniza texto usando tiktoken (BPE). Devuelve array de IDs."""
    if tiktoken is None:
        raise RuntimeError("tiktoken no disponible. pip install tiktoken")

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text, allowed_special=set())
    print(f"[INFO] Tokens generados: {len(tokens):,}")
    print(f"[INFO] Vocab size del tokenizador: {enc.n_vocab:,}")
    print(f"[INFO] Ratio de compresión: {len(text)/len(tokens):.1f} chars/token")
    return np.array(tokens, dtype=np.uint32)


def save_as_memmap(tokens: np.ndarray, output_dir: str, val_fraction: float = 0.1):
    """
    Guarda tokens como numpy memmap (archivos binarios mapeados en memoria).
    Split automático en train/val.
    """
    os.makedirs(output_dir, exist_ok=True)

    n = len(tokens)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val

    # Guardar train
    train_path = os.path.join(output_dir, "train.bin")
    train_mm = np.memmap(train_path, dtype=np.uint32, mode='w+', shape=(n_train,))
    train_mm[:] = tokens[:n_train]
    train_mm.flush()

    # Guardar val
    val_path = os.path.join(output_dir, "val.bin")
    val_mm = np.memmap(val_path, dtype=np.uint32, mode='w+', shape=(n_val,))
    val_mm[:] = tokens[n_train:]
    val_mm.flush()

    # Guardar metadata
    meta = {
        'total_tokens': n,
        'train_tokens': n_train,
        'val_tokens': n_val,
        'dtype': 'uint32',
    }
    import json
    with open(os.path.join(output_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n[INFO] Datos guardados en {output_dir}/")
    print(f"  -> train.bin: {n_train:,} tokens ({n_train * 4 / 1e6:.1f} MB)")
    print(f"  -> val.bin:   {n_val:,} tokens ({n_val * 4 / 1e6:.1f} MB)")

    return n_train, n_val


def download_resources(raw_dir: str):
    """Descarga recursos de texto si la carpeta está vacía."""
    if glob.glob(os.path.join(raw_dir, "*.txt")):
        print("[INFO] Ya hay archivos en data/raw/. Saltando descarga.")
        return

    os.makedirs(raw_dir, exist_ok=True)
    print("[INFO] Descargando recursos de texto...")

    urls = {
        "quijote.txt": "https://www.gutenberg.org/cache/epub/2000/pg2000.txt",
        "shakespeare.txt": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "linux_code.txt": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/linux/input.txt",
    }

    import urllib.request
    for name, url in urls.items():
        path = os.path.join(raw_dir, name)
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  -> Descargado: {name}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline de datos para LLM")
    parser.add_argument('--raw-dir', type=str, default='data/raw')
    parser.add_argument('--output-dir', type=str, default='data/processed')
    parser.add_argument('--encoding', type=str, default='cl100k_base')
    parser.add_argument('--download', action='store_true', help='Descargar recursos si no existen')
    args = parser.parse_args()

    if args.download:
        download_resources(args.raw_dir)

    print("=" * 60)
    print("📦 PIPELINE DE DATOS PARA LLM (tiktoken + memmap)")
    print("=" * 60)

    # 1. Recopilar texto
    print("\n[1/3] Recopilando texto crudo...")
    text = collect_raw_text(args.raw_dir)
    if not text:
        print("[ERROR] Sin texto. Usa --download o añade .txt a data/raw/")
        return

    # 2. Tokenizar
    print("\n[2/3] Tokenizando con tiktoken...")
    tokens = tokenize_text(text, args.encoding)

    # 3. Guardar
    print("\n[3/3] Guardando como memmap...")
    n_train, n_val = save_as_memmap(tokens, args.output_dir)

    # Estimaciones de entrenamiento
    tokens_per_sec_p100 = 50_000
    tokens_per_sec_l4 = 120_000
    print(f"\n⏱️  Tiempo estimado por epoch:")
    print(f"  -> Kaggle P100: ~{n_train/tokens_per_sec_p100/60:.0f} min")
    print(f"  -> NVIDIA L4:   ~{n_train/tokens_per_sec_l4/60:.0f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
