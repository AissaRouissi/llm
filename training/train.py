import os
import time
import argparse
import torch

# Importar dependencias del proyecto (ajustar path si se llama desde otra ruta)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gpt_config import config as base_config
from models.gpt_model import GPT
from training.lr_scheduler import get_lr

# Hiperparámetros del entrenamiento (production-ready)
GRAD_ACCUM_STEPS = 8 # Simular un batch mayor acumulando gradientes
MAX_ITERS = 10000
WARMUP_ITERS = 500
EVAL_INTERVAL = 500
EVAL_ITERS = 100
LOG_INTERVAL = 100
MAX_LR = base_config.learning_rate
WEIGHT_DECAY = 0.1

def setup_training(resume_path=None, compile_model=True):
    """Inicializa device, dtype, modelo, compilador, optimizer y scaler."""
    device = base_config.device
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    print(f"-> Target Device: {device} | Dtype: {dtype}")
    
    model = GPT(base_config)
    model.to(device)
    optimizer = model.configure_optimizers(WEIGHT_DECAY, MAX_LR, device)
    
    start_iter = 0
    if resume_path and os.path.exists(resume_path):
        print(f"-> Resumiendo desde: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt['iter_num']
        torch.set_rng_state(ckpt['rng_state']) # Reproducibilidad estricta
        
    if compile_model and device == 'cuda':
        print("-> Compilando modelo con torch.compile() para max performance...")
        model = torch.compile(model)
        
    try:
        scaler = torch.amp.GradScaler(device, enabled=(dtype == torch.float16))
    except (AttributeError, TypeError, RuntimeError):
        # Fallback para versiones de PyTorch < 2.4 o problemas con CPU
        class FakeScaler:
            def scale(self, loss): return loss
            def step(self, opt): opt.step()
            def update(self): pass
            def unscale_(self, opt): pass
        scaler = FakeScaler()
        
    return model, optimizer, scaler, start_iter, device, dtype

def get_batch(data_split: torch.Tensor, config) -> tuple[torch.Tensor, torch.Tensor]:
    """Genera un batch aleatorio, x=input, y=target desplazado 1 pos."""
    ix = torch.randint(len(data_split) - config.block_size, (config.batch_size,))
    x = torch.stack([data_split[i:i+config.block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+config.block_size+1] for i in ix])
    
    # Mover a VRAM asíncronamente para no bloquear CPU
    x = x.to(config.device, non_blocking=True)
    y = y.to(config.device, non_blocking=True)
    return x, y

@torch.no_grad()
def estimate_loss(model: torch.nn.Module, train_data, val_data, config) -> dict:
    """Evalúa la red en validación y entrenamiento de forma estocástica."""
    out = {}
    model.eval()
    splits = {'train': train_data, 'val': val_data}
    
    for split, data in splits.items():
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(data, config)
            # Evaluar siempre usando precision mixta (bfloat16) si es posible
            with torch.amp.autocast(device_type=config.device, dtype=torch.bfloat16 if config.device == 'cuda' else torch.float32):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
        
    model.train()
    return out

def train_loop(model, optimizer, scaler, start_iter, train_data, val_data, config, device, dtype):
    """Bucle principal optimizado para GPU (grad accum, mixed precision, lr decay)."""
    os.makedirs('checkpoints', exist_ok=True)
    best_val_loss = float('inf')
    t0 = time.time()
    
    for iter_num in range(start_iter, MAX_ITERS):
        # 1. Ajustar Learning Rate
        lr = get_lr(iter_num, WARMUP_ITERS, MAX_ITERS, MAX_LR)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # 2. Evaluación periódica y Guardado de Checkpoint
        if iter_num > start_iter and iter_num % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data, config)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'config': config,
                'val_loss': losses['val'],
                'rng_state': torch.get_rng_state()
            }
            torch.save(checkpoint, f"checkpoints/ckpt_last.pt")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(checkpoint, f"checkpoints/ckpt_best.pt")
            
            # Generar texto rápido para ver progreso
            if iter_num % (EVAL_INTERVAL * 2) == 0:
                print("-> Generando muestra...")
                model.eval()
                ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
                gen = model.generate(ctx, max_new_tokens=30)
                print(f"-> Output raw (tokens): {gen[0].tolist()}")
                model.train()

        # 3. Micro-steps (Gradient Accumulation)
        for micro_step in range(GRAD_ACCUM_STEPS):
            X, Y = get_batch(train_data, config)
            
            # Autocast: El forward y la loss calculation en bfloat16
            with torch.amp.autocast(device_type=device, dtype=dtype):
                logits, loss = model(X, Y)
                loss = loss / GRAD_ACCUM_STEPS # Normalizar para que math cuadre
            
            # Backward: Escalar gradientes si usamos float16 (en bf16 scaler.scale hace nada)
            scaler.scale(loss).backward()
            
        # 4. Actualización y Step del Optimizador
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True) # Más eficiente que optim.zero_grad()
        
        # 5. Tiempos y Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % LOG_INTERVAL == 0:
            tok_per_sec = (GRAD_ACCUM_STEPS * config.batch_size * config.block_size) / dt
            loss_val = loss.item() * GRAD_ACCUM_STEPS
            vram_gb = torch.cuda.memory_allocated() / (1024**3) if device == 'cuda' else 0.0
            vram_str = f"{vram_gb:.1f}GB" if device == 'cuda' else "N/A"
            
            print(f"[STEP {iter_num:4d}] loss={loss_val:.4f} | val={best_val_loss:.4f} | lr={lr:.2e} | tok/s={tok_per_sec:.0f} | VRAM={vram_str}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint file')
    parser.add_argument('--data', type=str, default='data/processed/data.pt', help='Path to tokenized data')
    parser.add_argument('--compile', action='store_true', default=False, help='Force torch.compile (recommended for L4)')
    args = parser.parse_args()
    
    print("--- INICIANDO FASE DE ENTRENAMIENTO ---")
    # Configuración por defecto: compile activo si hay CUDA y args.compile fue omitido, pero dejemos el control al usuario
    use_compile = True if torch.cuda.is_available() else args.compile
    
    model, optimizer, scaler, start_iter, device, dtype = setup_training(args.resume, compile_model=use_compile)
    
    if not os.path.exists(args.data):
        print(f"ADVERTENCIA: No se encontró {args.data}. Creando tensor de pruebas...")
        data = torch.randint(0, base_config.vocab_size, (5000,))
    else:
        print("Cargando dataset pre-tokenizado...")
        data = torch.load(args.data)
        
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Dataset Split -> Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")
    
    train_loop(model, optimizer, scaler, start_iter, train_data, val_data, base_config, device, dtype)

if __name__ == '__main__':
    main()
