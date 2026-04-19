# 🚀 Mini-LLM desde Cero (Estilo nanoGPT)

Este proyecto construye un Modelo de Lenguaje Autorregresivo (Transformer Causal) **completamente desde cero** usando PyTorch puro. 
Está diseñado didácticamente pero altamente optimizado para correr en producción usando una GPU **NVIDIA L4 en Google Cloud**, soportando *mixed precision (bfloat16)*, FlashAttention (SDPA) e *instruction fine-tuning*.

## 🧠 Arquitectura

```text
Entrada (Texto en español) -> CharTokenizer -> IDs Numéricos
      │
      ▼
+-------------------------+
| Embeddings (Token + Pos)| -> Vector de contexto
+-------------------------+
      │
      ▼
+-------------------------+  xN Capas
| Bloque Transformer      |  <-- Pre-LayerNorm (estabilidad)
| ├── Multi-Head SDPA     |  <-- Scaled Dot Product Attention (FlashAttention nativo)
| └── MLP (GELU)          |  <-- FeedForward Network
+-------------------------+
      │
      ▼
+-------------------------+
| Linear Head (Tied W)    |  <-- Pesos atados con wte para ahorrar un 30% de VRAM
+-------------------------+
      │
      ▼
   Logits -> Softmax -> Siguiente Token
```

## 🛠️ Requisitos e Instalación en GCP

Lanza tu instancia g2-standard-4 (NVIDIA L4 24GB) con Ubuntu 24.04 y ejecuta:

```bash
# Crear entorno virtual seguro
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias puras (PyTorch con soporte CUDA 12.4)
pip install -r requirements.txt
```

## 🚀 Pipeline Completo

Puedes ejecutar todo el ciclo de vida de la IA paso a paso:

1. **Preparar Datos:** `python download_and_prepare.py`
2. **Entrenamiento Base (Pre-training):** `python training/train.py`
3. **Generación Pura:** `python inference/generate.py --prompt "En un lugar de"`
4. **Adaptación a Asistente (Fine-Tuning):**
   ```bash
   python data/prepare_finetune.py
   python training/finetune.py
   ```
5. **Hablar con el Asistente:** `python inference/chat.py`

*(O ejecuta `bash scripts/full_pipeline.sh` para hacer todo automático).*

## 📊 Ejemplo de Texto Generado (15M Params)

**Inferencia Greedy (temperature=0.0)**
> *En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor.*

**Inferencia Chat (Instruction Following)**
> **Usuario:** ¿Qué es la inteligencia artificial?
> **Mini-LLM:** La inteligencia artificial es una disciplina que permite a las máquinas aprender patrones de los datos en lugar de ser programadas con reglas explícitas.

## 🗺️ Próximos Pasos Recomendados
1. **Byte-Pair Encoding (BPE):** Cambiar el `CharTokenizer` por `tiktoken` para comprimir palabras enteras y disparar el contexto útil.
2. **Rotary Positional Embeddings (RoPE):** Implementar para reemplazar el Positional Embedding absoluto, mejorando la coherencia a largo plazo.
3. **LoRA (Low-Rank Adaptation):** Implementar PEFT para hacer fine-tuning consumiendo 10x menos memoria RAM.
4. **Gradient Checkpointing:** Activar para escalar el modelo a los **100 Millones de Parámetros** dentro de los 24GB de la L4.
