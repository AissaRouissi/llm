import os
import sys
import time
import argparse
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.generate import load_model_from_checkpoint
from training.lr_scheduler import get_lr
from training.train import get_batch

def estimate_loss(model, train_data, val_data, config, iters=50):
    """Evaluación específica para finetuning considerando el diccionario {'x', 'y'}."""
    out = {}
    model.eval()
    splits = {'train': train_data, 'val': val_data}
    
    for split, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            # En finetune guardamos tensores separados x e y por el masking
            ix = torch.randint(len(data['x']) - config.block_size, (config.batch_size,))
            X = torch.stack([data['x'][i:i+config.block_size] for i in ix])
            Y = torch.stack([data['y'][i:i+config.block_size] for i in ix])
            X, Y = X.to(config.device), Y.to(config.device)
            
            with torch.amp.autocast(device_type=config.device, dtype=torch.bfloat16 if config.device == 'cuda' else torch.float32):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
        
    model.train()
    return out

def finetune_loop(model, optimizer, scaler, train_data, val_data, config, args, device, dtype):
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    early_stopping_counter = 0
    t0 = time.time()
    
    for iter_num in range(args.iters):
        lr = get_lr(iter_num, warmup_iters=100, max_iters=args.iters, max_lr=args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        if iter_num > 0 and iter_num % 200 == 0:
            losses = estimate_loss(model, train_data, val_data, config)
            print(f"[FT STEP {iter_num:4d}] val_loss={losses['val']:.4f} | train_loss={losses['train']:.4f}")
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                early_stopping_counter = 0
                checkpoint = {
                    'model': model.state_dict(),
                    'config': config,
                    'val_loss': losses['val'],
                }
                torch.save(checkpoint, os.path.join(args.output_dir, "finetuned_best.pt"))
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= 3:
                    print("🛑 Early stopping: val_loss no mejora después de 3 evaluaciones consecutivas.")
                    break

        # Grad accum
        for micro_step in range(4): # 4 steps accum por default en finetune
            ix = torch.randint(len(train_data['x']) - config.block_size, (config.batch_size,))
            X = torch.stack([train_data['x'][i:i+config.block_size] for i in ix])
            Y = torch.stack([train_data['y'][i:i+config.block_size] for i in ix])
            X, Y = X.to(device), Y.to(device)
            
            with torch.amp.autocast(device_type=device, dtype=dtype):
                logits, loss = model(X, Y)
                loss = loss / 4 
            scaler.scale(loss).backward()
            
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % 50 == 0:
            print(f"[FT STEP {iter_num:4d}] loss={loss.item()*4:.4f} | lr={lr:.2e} | dt={dt*1000:.0f}ms")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-checkpoint', type=str, default='checkpoints/ckpt_best.pt')
    parser.add_argument('--output-dir', type=str, default='checkpoints/finetune')
    parser.add_argument('--lr', type=float, default=3e-5) # LR ~10x menor que pre-training
    parser.add_argument('--iters', type=int, default=2000)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    print(f"Cargando modelo base de {args.base_checkpoint}...")
    model, config = load_model_from_checkpoint(args.base_checkpoint, device)
    
    # Optimizador con weight decay estándar
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=args.lr, device_type=device)
    
    try:
        scaler = torch.amp.GradScaler(device, enabled=(dtype == torch.float16))
    except (AttributeError, TypeError, RuntimeError):
        class FakeScaler:
            def scale(self, loss): return loss
            def step(self, opt): opt.step()
            def update(self): pass
            def unscale_(self, opt): pass
        scaler = FakeScaler()
        
    print("Cargando dataset de finetuning...")
    train_data = torch.load("data/processed/finetune_train.pt")
    val_data = torch.load("data/processed/finetune_val.pt")
    
    print(f"Iniciando Fine-Tuning Completo ({args.iters} iters)...")
    model.train()
    finetune_loop(model, optimizer, scaler, train_data, val_data, config, args, device, dtype)
    print("✅ Fine-tuning finalizado.")

if __name__ == '__main__':
    main()
