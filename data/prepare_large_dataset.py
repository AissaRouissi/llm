import os
import argparse
import urllib.request
import re

def download_gutenberg(urls, output_file):
    print("Descargando libros de Project Gutenberg en español...")
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for url in urls:
            print(f"-> Extrayendo {url}...")
            try:
                response = urllib.request.urlopen(url)
                text = response.read().decode('utf-8')
                
                # Limpieza agresiva de metadatos de Gutenberg (opcional/simplificado)
                text = re.sub(r'(\r\n|\r|\n)+', '\n', text) # Normalizar saltos de línea
                
                # Escribir el corpus puro
                out_f.write(text)
                out_f.write('\n\n')
            except Exception as e:
                print(f"Error descargando {url}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, choices=['shakespeare', 'gutenberg', 'custom'], default='gutenberg')
    args = parser.parse_args()
    
    output_raw = "data/raw/large_dataset.txt"
    os.makedirs("data/raw", exist_ok=True)
    
    if args.source == 'gutenberg':
        # 5 Clásicos en dominio público en español para entrenar la sintaxis
        urls = [
            "https://www.gutenberg.org/cache/epub/2000/pg2000.txt",   # Don Quijote
            "https://www.gutenberg.org/cache/epub/15502/pg15502.txt", # Fortunata y Jacinta
            "https://www.gutenberg.org/cache/epub/44583/pg44583.txt", # Niebla (Unamuno)
            "https://www.gutenberg.org/cache/epub/55269/pg55269.txt", # La Regenta
            "https://www.gutenberg.org/cache/epub/14064/pg14064.txt"  # Poesías de Rosalía de Castro
        ]
        download_gutenberg(urls, output_raw)
    else:
        print(f"Fuente seleccionada: {args.source}. Escribiendo archivo mock para test.")
        with open(output_raw, 'w', encoding='utf-8') as f:
            f.write("Este es un dataset de ejemplo para pruebas de pipeline.\n")
            
    # Estadísticas post-descarga
    print("\n" + "="*50)
    print("📦 ESTADÍSTICAS DEL DATASET GIGANTE")
    print("="*50)
    size_mb = os.path.getsize(output_raw) / (1024*1024)
    tokens_est = int(size_mb * 1024 * 1024) # 1 char ≈ 1 token en CharTokenizer
    
    print(f"-> Archivo unificado guardado en: {output_raw}")
    print(f"-> Tamaño físico: {size_mb:.2f} MB")
    print(f"-> Tokens estimados: {tokens_est:,} (CharTokenizer)")
    
    # 1 L4 GPU procesa ~100k tokens por segundo en bfloat16
    tokens_per_sec_l4 = 100_000 
    seconds = tokens_est / tokens_per_sec_l4
    minutes = seconds / 60
    
    print(f"-> ⏱️ Tiempo estimado de procesamiento (1 Epoch) en NVIDIA L4: ~{minutes:.1f} minutos")
    print(f"-> 💸 Coste estimado en GCP ($0.71/h): ~${(minutes/60)*0.71:.3f}")
    print("="*50)

if __name__ == "__main__":
    main()
