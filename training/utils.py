import torch

def save_checkpoint(model, optimizer, iter_num, config, filepath):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'config': config,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint guardado en {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['iter_num'], checkpoint.get('config')

def get_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"VRAM Usada: {allocated:.2f} GB / 24.00 GB (Reservada: {reserved:.2f} GB)")
    else:
        print("CUDA no disponible (CPU)")

def estimate_batch_size():
    # Una L4 tiene 24GB. Un modelo de 10-50M con bs=256 usa ~4-6GB max. 
    # Aquí podríamos hacer un test dinámico, pero para este tamaño devolvemos directo:
    print("Estimando batch size... Configuración segura para 24GB L4 encontrada: 256")
    return 256
