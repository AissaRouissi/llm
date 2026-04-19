import os
import json

def create_ascii_bar_chart(data, width=50):
    if not data: return "Sin datos"
    min_val, max_val = min(data), max(data)
    range_val = max_val - min_val if max_val > min_val else 1
    
    chart = []
    for i, val in enumerate(data):
        normalized = int(((val - min_val) / range_val) * width)
        bar = '█' * normalized + '░' * (width - normalized)
        chart.append(f"Step {i*100:4d} | {val:.4f} | {bar}")
    return "\n".join(chart)

def analyze_logs(log_path):
    steps, train_loss, val_loss = [], [], []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                steps.append(d['step'])
                train_loss.append(d.get('loss', d.get('train_loss', 0)))
                val_loss.append(d.get('val_loss', 0))
    except FileNotFoundError:
        print("[AVISO] No se encontró log de entrenamiento real. Cargando datos de simulación.")
        steps = list(range(0, 1000, 100))
        train_loss = [4.5, 3.2, 2.5, 2.1, 1.8, 1.6, 1.5, 1.4, 1.35, 1.3]
        val_loss = [4.5, 3.4, 2.8, 2.5, 2.3, 2.2, 2.2, 2.3, 2.4, 2.6]
        
    print("\n" + "="*60)
    print("📈 ANÁLISIS DE CURVA DE ENTRENAMIENTO (ASCII)")
    print("="*60)
    
    print("\nEvolución del Val Loss:")
    # Muestrear máx 15 puntos para que encaje en terminales estándar
    sample_rate = max(1, len(val_loss) // 15)
    sampled_val = val_loss[::sample_rate]
    print(create_ascii_bar_chart(sampled_val))
    
    print("\n--- 🩺 DIAGNÓSTICO AUTOMÁTICO ---")
    final_train, final_val, best_val = train_loss[-1], val_loss[-1], min(val_loss)
    
    overfitting = final_val > best_val * 1.15
    print(f"✓ Overfitting Severo: {'🚨 SÍ' if overfitting else '✅ NO'}")
    
    spikes = any(val_loss[i] > val_loss[i-1] * 1.5 for i in range(1, len(val_loss)))
    print(f"✓ Loss Spikes (LR alto): {'🚨 SÍ' if spikes else '✅ NO'}")
    
    print("\n🤖 RECOMENDACIÓN DEL ARQUITECTO:")
    if overfitting:
        print("-> PARAR. El modelo está memorizando el dataset sin aprender lógica universal.")
        print("-> Siguiente paso: Incrementar Weight Decay al 0.2 o conseguir más datos (Fase 6).")
    elif spikes:
        print("-> BAJAR LEARNING RATE. La optimización explotó numéricamente (Gradient Explosion).")
    else:
        print("-> CONTINUAR. Convergencia sana observada. El gradiente fluye suavemente.")
        print("-> Siguiente paso: Fase 5 (Fine-tuning de instrucción) o Escalar arquitectura.")
        
    with open("evaluation/training_report.txt", "w") as f:
        f.write("--- LLM Training Health Report ---\n")
        f.write(f"Steps evaluados: {len(steps)}\n")
        f.write(f"Loss Final (Train): {final_train:.4f}\n")
        f.write(f"Loss Final (Val): {final_val:.4f}\n")
        f.write(f"Loss Mínima Histórica: {best_val:.4f}\n")
        f.write(f"Overfitting detectado: {overfitting}\n")
        f.write(f"Instability detectada: {spikes}\n")
    print("\n[Reporte guardado en evaluation/training_report.txt]\n")

if __name__ == "__main__":
    analyze_logs("training_logs.jsonl")
