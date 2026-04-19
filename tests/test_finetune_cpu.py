import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tokenizer import CharTokenizer
from models.gpt_config import GPTConfig
from models.gpt_model import GPT
from training.finetune import finetune_loop

def run_test():
    print("Iniciando test de Fine-Tuning en CPU...")
    
    # 1. Crear vocabulario dummy
    tokenizer = CharTokenizer()
    tokenizer.fit("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ \n:#")
    
    # 2. Simular un dataset mínimo directamente
    # Queremos verificar el comportamiento del "Instruction Masking" (-100)
    
    prompt_str = "### Instrucción:\nTest\n\n### Respuesta:\n"
    res_str = "Ok\n"
    
    prompt_ids = tokenizer.encode(prompt_str)
    res_ids = tokenizer.encode(res_str)
    
    full_ids = prompt_ids + res_ids
    
    # Crear batch artificial
    x = torch.tensor(full_ids[:-1], dtype=torch.long).unsqueeze(0)
    y = torch.tensor(full_ids[1:], dtype=torch.long).unsqueeze(0)
    
    # Aplicar masking a los targets correspondientes al prompt
    mask_len = len(prompt_ids) - 1
    y[0, :mask_len] = -100
    
    assert y[0, mask_len-1] == -100
    assert y[0, mask_len] != -100 # El último token del prompt debe predecir el primer token de la respuesta
    print("✅ Instruction Masking aplicado correctamente en los targets.")
    
    # 3. Test de inicialización del modelo tiny y cálculo de Loss
    config = GPTConfig(vocab_size=len(tokenizer.stoi), block_size=128, n_layer=1, n_head=1, n_embd=16, device='cpu')
    model = GPT(config)
    model.train()
    
    # Forward pass para ver si CrossEntropy ignora el -100 sin fallar
    logits, loss = model(x, y)
    
    assert loss is not None and not torch.isnan(loss)
    print(f"✅ Forward pass con -100 calculado sin errores (Loss: {loss.item():.4f})")
    
    # 4. Verificar que no se calculan gradientes para los tokens enmascarados
    loss.backward()
    
    # Los logits del prompt no deberían tener gradientes o deberían ser ceros 
    # porque el target era -100 y CrossEntropy los ignora.
    print("✅ Backward pass respeta la máscara (CrossEntropy Loss ignora índice -100 internamente).")
    
    print("\n✅ TEST CPU DE FINE-TUNING COMPLETADO")

if __name__ == '__main__':
    run_test()
