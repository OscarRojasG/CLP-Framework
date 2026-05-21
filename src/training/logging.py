import json
import os

def save_experiment_config(filepath, config):
    data = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try: data = json.load(f)
            except: pass
    
    # Guardamos la configuración en una llave raíz dedicada
    data["config"] = config
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def save_phase_history(filepath, phase, history):
    """Guarda el historial de una fase completa en un JSON existente."""
    data = {}
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try: data = json.load(f)
            except: pass

    # Estructura: phase_X -> epoch_Y -> {train/val} -> {losses/metrics}
    data[f"phase_{phase}"] = {f"epoch_{e}": stats for e, stats in history.items()}

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)