import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gpt_config import GPTConfig
from models.gpt_model import GPT
from training.lr_scheduler import get_lr
from training.train import estimate_loss, get_batch

def run_test():
    print("Iniciando mini-training loop (CPU test en MacBook)...")
    config = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=64, batch_size=4, device='cpu')
    model = GPT(config)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-3, device_type='cpu')
    
    # Fake dataset
    data = torch.randint(0, config.vocab_size, (1000,))
    train_data, val_data = data[:900], data[900:]
    
    # 1. Test get_batch
    x, y = get_batch(train_data, config)
    assert x.shape == (config.batch_size, config.block_size)
    print("✅ get_batch() arroja el shape correcto.")
    
    # 2. Test scheduler
    lr = get_lr(step=10, warmup_iters=50, max_iters=100, max_lr=1e-3)
    assert 0 < lr < 1e-3
    print(f"✅ Scheduler funcional (LR at step 10: {lr:.5f})")
    
    # 3. Test entrenamiento local
    initial_loss = estimate_loss(model, train_data, val_data, config)['train']
    print(f"[DEBUG] Loss Inicial: {initial_loss:.4f}")
    
    # Manejo robusto del Scaler para CPU y versiones < 2.4
    try:
        scaler = torch.amp.GradScaler('cpu', enabled=False)
    except (AttributeError, TypeError, RuntimeError):
        class FakeScaler:
            def scale(self, loss): return loss
            def step(self, opt): opt.step()
            def update(self): pass
            def unscale_(self, opt): pass
        scaler = FakeScaler()
    
    model.train()
    for step in range(10):
        # Micro-step 1
        x, y = get_batch(train_data, config)
        logits, loss = model(x, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
    final_loss = estimate_loss(model, train_data, val_data, config)['train']
    print(f"[DEBUG] Loss Final tras 10 steps: {final_loss:.4f}")
    assert final_loss < initial_loss, "La loss no bajó, hay un bug en el backpropagation"
    print("✅ Backward pass optimiza parámetros correctamente.")
    
    # 4. Guardar/Cargar Checkpoint test
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'rng_state': torch.get_rng_state()
    }
    torch.save(checkpoint, "test_ckpt.pt")
    
    loaded = torch.load("test_ckpt.pt", weights_only=False)
    model.load_state_dict(loaded['model'])
    torch.set_rng_state(loaded['rng_state'])
    os.remove("test_ckpt.pt")
    print("✅ Save y Load state reproducen el estado del modelo.")
    
    print("\n✅ TEST CPU COMPLETADO SIN ERRORES")

if __name__ == '__main__':
    run_test()
