import math

def get_lr(step: int, warmup_iters: int, max_iters: int, max_lr: float) -> float:
    """
    Cosine learning rate scheduler con warmup lineal.
    Implementación nativa sin dependencias de torch.optim.lr_scheduler para mayor control.
    """
    min_lr = max_lr * 0.1
    
    # 1) Fase de warmup: crecimiento lineal del LR
    if step < warmup_iters:
        return max_lr * (step + 1) / warmup_iters
        
    # 2) Pasado max_iters: mantenemos min_lr
    if step > max_iters:
        return min_lr
        
    # 3) Cosine decay: curva suave desde max_lr hasta min_lr
    decay_ratio = (step - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
