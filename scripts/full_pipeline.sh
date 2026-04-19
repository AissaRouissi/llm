#!/bin/bash
set -e # Abortar al primer error detectado

echo "================================================="
echo "🚀 INICIANDO FULL PIPELINE MINI-LLM EN GOOGLE CLOUD"
echo "================================================="

echo -e "\n[1/6] Preparando Entorno y Descargando Datos Base..."
python download_and_prepare.py

echo -e "\n[2/6] Ejecutando Tests de Integridad (Matemáticas CPU)..."
python tests/test_model.py
python tests/test_training_cpu.py
python tests/test_generate_cpu.py
python tests/test_finetune_cpu.py

echo -e "\n[3/6] Iniciando Pre-Training Base (GPT Causal)..."
echo "INFO: En GCP L4 esto aprovechará FlashAttention y bfloat16."
python training/train.py --iters 500

echo -e "\n[4/6] Extracción y Evaluación de Perplejidad..."
python evaluation/perplexity.py
python evaluation/analyze_training.py

echo -e "\n[5/6] Instruction Fine-Tuning (Modo Chatbot)..."
python data/prepare_finetune.py
python training/finetune.py --iters 200

echo -e "\n[6/6] Comparación de Estrategias de Generación..."
python inference/compare_sampling.py

echo -e "\n================================================="
echo "✅ PIPELINE COMPLETADO EXITOSAMENTE"
echo "-> Inicia tu asistente interactivo ejecutando:"
echo "   python inference/chat.py"
echo "================================================="
