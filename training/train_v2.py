"""
Training Loop Production-Grade para LLaMA desde cero.

Características:
- Mixed Precision automático (bfloat16 en L4/T4, float16+GradScaler en P100)
- Gradient Checkpointing (entrenar modelos 2x más grandes en la misma VRAM)
- Cosine Decay LR con Warmup
- Checkpointing tolerante a fallos (sobrevive a desconexiones de Kaggle)
- Monitoreo opcional con wandb
- torch.compile para máximo throughput
"""
import os
import sys
import math
import time
import json
import argparse
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llama_config import LLaMAConfig, get_small_config, get_medium_config
from models.llama_model import LLaMA


# ==============================================================================
# Funciones auxiliares
# ==============================================================================
def get_lr(it, warmup_iters, max_iters, max_lr, min_lr_ratio=0.1):
    """Cosine Decay con Warmup lineal."""
    min_lr = max_lr * min_lr_ratio
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it >= max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def get_batch(data_mm, batch_size, block_size, device):
    """Genera un batch aleatorio desde memmap (lectura directa de disco)."""
    ix = torch.randint(len(data_mm) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data_mm[i:i+block_size].astype(np.int64).copy()) for i in ix])
    y = torch.stack([torch.from_numpy(data_mm[i+1:i+1+block_size].astype(np.int64).copy()) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, train_mm, val_mm, config, eval_iters, device, dtype, use_gc):
    """Evalúa loss en train y val (estocástico)."""
    model.eval()
    out = {}
    for split, data in [('train', train_mm), ('val', val_mm)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, config.batch_size, config.max_seq_len, device)
            with torch.amp.autocast(device_type=device if device != 'mps' else 'cpu', dtype=dtype):
                _, loss = model(X, Y, use_gradient_checkpointing=use_gc)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(model, optimizer, iter_num, val_loss, config, path):
    """Guarda checkpoint tolerante a fallos."""
    tmp_path = path + ".tmp"
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'val_loss': val_loss,
        'config': config,
        'rng_state': torch.get_rng_state(),
    }
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)  # Atómico: si Kaggle se cierra a mitad, no corrompe el archivo


def load_checkpoint(path, model, optimizer, device):
    """Carga checkpoint para resumir entrenamiento."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    torch.set_rng_state(ckpt['rng_state'])
    return ckpt['iter_num'], ckpt.get('val_loss', float('inf'))


# ==============================================================================
# Training Loop Principal
# ==============================================================================
def train(args):
    # --- 1. Auto-detección de hardware ---
    if torch.cuda.is_available():
        device = 'cuda'
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            use_scaler = False
            print(f"-> GPU: {torch.cuda.get_device_name()} | Usando bfloat16 (óptimo)")
        else:
            dtype = torch.float16
            use_scaler = True
            print(f"-> GPU: {torch.cuda.get_device_name()} | Usando float16 + GradScaler")
    else:
        device = 'cpu'
        dtype = torch.float32
        use_scaler = False
        print("-> CPU mode (solo para testing)")

    # --- 2. Configuración del modelo ---
    if args.size == 'small':
        config = get_small_config(device=device)
    else:
        config = get_medium_config(device=device)

    # Cargar metadata del dataset para ajustar vocab_size
    meta_path = os.path.join(args.data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"-> Dataset: {meta['train_tokens']:,} train tokens, {meta['val_tokens']:,} val tokens")

    # tiktoken cl100k_base tiene vocab_size=100277
    config.vocab_size = 100277
    config.batch_size = args.batch_size

    print(f"-> Modelo: dim={config.dim}, layers={config.n_layers}, heads={config.n_heads}, kv_heads={config.n_kv_heads}")

    # --- 3. Crear modelo ---
    model = LLaMA(config)
    model.to(device)
    n_params = model.get_num_params()
    print(f"-> Parámetros: {n_params:,} ({n_params/1e6:.1f}M)")

    # --- 4. Optimizer ---
    optimizer = model.configure_optimizers(args.weight_decay, args.lr, device)

    # --- 5. GradScaler (solo para float16) ---
    if use_scaler:
        scaler = torch.amp.GradScaler('cuda')
    else:
        class NoScaler:
            def scale(self, loss): return loss
            def step(self, opt): opt.step()
            def update(self): pass
            def unscale_(self, opt): pass
        scaler = NoScaler()

    # --- 6. Compilar modelo (si GPU compatible) ---
    start_iter = 0
    best_val_loss = float('inf')

    if args.resume and os.path.exists(args.resume):
        print(f"-> Resumiendo desde: {args.resume}")
        start_iter, best_val_loss = load_checkpoint(args.resume, model, optimizer, device)
        print(f"   Iteración: {start_iter}, Mejor val_loss: {best_val_loss:.4f}")

    if args.compile and device == 'cuda':
        print("-> Compilando modelo con torch.compile (max-autotune)...")
        model = torch.compile(model, mode='max-autotune')

    # --- 7. Cargar datos (memmap) ---
    train_mm = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint32, mode='r')
    val_mm = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint32, mode='r')

    # --- 8. Wandb (opcional) ---
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="mini-llm", config=vars(config))
        except ImportError:
            print("[AVISO] wandb no instalado. Continuando sin monitoreo.")
            use_wandb = False

    # --- 9. TRAINING LOOP ---
    print(f"\n{'='*60}")
    print(f"🔥 ENTRENAMIENTO INICIADO ({args.max_iters} iteraciones)")
    print(f"{'='*60}\n")

    model.train()
    t0 = time.time()
    log_file = open("training_log.jsonl", "a")

    for iter_num in range(start_iter, args.max_iters):
        # LR Schedule
        lr = get_lr(iter_num, args.warmup_iters, args.max_iters, args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Evaluación periódica
        if iter_num > start_iter and iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, train_mm, val_mm, config, args.eval_iters, device, dtype, args.grad_checkpoint)
            print(f"[EVAL {iter_num:5d}] train={losses['train']:.4f} | val={losses['val']:.4f} | best={best_val_loss:.4f}")

            # Log
            log_entry = {"step": iter_num, "train_loss": losses['train'], "val_loss": losses['val'], "lr": lr}
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            if use_wandb:
                import wandb
                wandb.log(log_entry)

            # Guardar mejor checkpoint
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_checkpoint(model, optimizer, iter_num, best_val_loss, config, "checkpoints/best.pt")
                print(f"  -> 🏆 Nuevo mejor modelo guardado!")

            # Guardar último (para resume)
            save_checkpoint(model, optimizer, iter_num, losses['val'], config, "checkpoints/last.pt")

        # Gradient Accumulation
        for micro in range(args.grad_accum_steps):
            X, Y = get_batch(train_mm, config.batch_size, config.max_seq_len, device)
            with torch.amp.autocast(device_type=device if device != 'mps' else 'cpu', dtype=dtype):
                _, loss = model(X, Y, use_gradient_checkpointing=args.grad_checkpoint)
                loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % args.log_interval == 0:
            loss_val = loss.item() * args.grad_accum_steps
            tps = (args.grad_accum_steps * config.batch_size * config.max_seq_len) / dt
            vram = f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if device == 'cuda' else "N/A"
            print(f"[STEP {iter_num:5d}] loss={loss_val:.4f} | lr={lr:.2e} | tok/s={tps:.0f} | VRAM={vram}")

    log_file.close()
    print(f"\n✅ Entrenamiento completado. Mejor val_loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train LLaMA-style LLM")
    parser.add_argument('--size', type=str, default='small', choices=['small', 'medium'])
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--resume', type=str, default='checkpoints/last.pt')
    parser.add_argument('--max-iters', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup-iters', type=int, default=200)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--eval-interval', type=int, default=250)
    parser.add_argument('--eval-iters', type=int, default=50)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--grad-checkpoint', action='store_true', default=False, help='Activar Gradient Checkpointing')
    parser.add_argument('--compile', action='store_true', default=False, help='Activar torch.compile')
    parser.add_argument('--wandb', action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
