# %% [markdown]
# # 🔭 Análisis Exploratorio y Evaluación del Mini-LLM
# Este script compatible con VS Code y Jupyter te permite bucear
# en los pesos e intenciones matemáticas de tu Transformer recién nacido.

# %%
import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tokenizer import CharTokenizer
from models.gpt_config import GPTConfig
from models.gpt_model import GPT
from evaluation.perplexity import calculate_perplexity

print("Entorno cargado. Listo para análisis forense del LLM.")

# %% [markdown]
# ## 1. Disección del Vocabulario
# %%
tokenizer = CharTokenizer()
vocab_path = "data/processed/vocab.json"
if os.path.exists(vocab_path):
    tokenizer.load(vocab_path)
else:
    tokenizer.fit("abcdefghijklmnopqrstuvwxyz \n")
    
print(f"Tamaño total del vocabulario (vocab_size): {len(tokenizer.stoi)}")
print("Primeros 10 caracteres mapeados:")
print(list(tokenizer.stoi.items())[:10])

# %% [markdown]
# ## 2. Test Inferencia Dinámica
# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = GPTConfig(vocab_size=len(tokenizer.stoi), block_size=64, n_layer=2, n_head=2, n_embd=32, device=device)
model = GPT(config)
model.to(device)
model.eval()

prompt = "¿"
print(f"Generando texto a partir del prompt: '{prompt}'")
idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
with torch.no_grad():
    out = model.generate(idx, max_new_tokens=40, temperature=0.9)
print("Resultado:")
print(tokenizer.decode(out[0].tolist()))

# %% [markdown]
# ## 3. Análisis de Mecanismos de Atención (El 'Corazón' del Transformer)
# Visualizamos a qué tokens presta atención el último token predicho.
# %%
def plot_attention_ascii(attn_weights, tokens):
    print("\n🗺️ MAPA DE ATENCIÓN (Heatmap ASCII)")
    print("Muestra cuánto peso (importancia) le dio la red neuronal a cada letra del pasado\n")
    
    last_word_attn = attn_weights[-1, :].tolist()
    
    for i, tok in enumerate(tokens):
        # Mapeamos probabilidad 0-1 a 20 bloques ASCII
        val = last_word_attn[i]
        bars = int(val * 30)
        # Escapamos el salto de línea para que se imprima bien
        display_tok = repr(tok) if tok == '\n' else tok
        print(f"{display_tok:4s} | {'█' * bars} {val:.3f}")

# Simulamos la matriz de atención (ya que no la guardamos explícitamente en el forward() por eficiencia)
text = "el perro verde"
tokens = list(text)
T = len(tokens)

# Creamos una matriz causal triangular falsa para demostración visual
dummy_attn = torch.tril(torch.rand(T, T))
# Hacemos que la palabra 'perro' sea muy importante para predecir 'verde'
dummy_attn[-1, 3:8] += 2.0 
dummy_attn = F.normalize(dummy_attn, p=1, dim=-1)

plot_attention_ascii(dummy_attn, tokens)

# %% [markdown]
# ## 4. Conclusión
# Si has ejecutado todo esto: ¡Felicidades! Acabas de entender el ciclo
# de vida completo de un modelo de Inteligencia Artificial Generativa.
