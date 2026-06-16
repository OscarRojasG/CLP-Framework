from settings import OUTPUT_FOLDER
from data.generation import read_output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from models.base.transformer import Transformer
from data.preprocessing import load_dataset
from torch.utils.data import DataLoader
import torch
from IPython.display import display
import numpy as np

def get_vcs_ranking_frequencies(filename: str):
    data = load_data(filename)
    y = np.array(data.Y)
    counter = {}
        
    for i in range(len(y)):
        max_y_val = np.max(y[i])
        best_action_idx = np.where(y[i] == max_y_val)[0][0]
        best_action_idx = int(best_action_idx)
        counter[best_action_idx] = counter.get(best_action_idx, 0) + 1

    # Generar el arreglo de frecuencias acumuladas
    if not counter:
        return []

    max_pos = max(counter.keys())
    arr_counter = [counter.get(i, 0) for i in range(max_pos + 1)]

    return arr_counter

def vcs_comparison(model, filename, metrics, num_actions, cuts):
    data = load_dataset(filename)

    data._open_file()
    try:
        ranks = data.file['ranks'][:]
    finally:
        data.close()

    # Guardamos el total de muestras para calcular porcentajes luego
    total_samples = len(ranks)

    # 1. Generación de predicciones globales
    y = np.eye(num_actions)[ranks]

    preds_vcs = np.zeros((total_samples, num_actions), dtype=int)
    preds_vcs[:] = np.arange(num_actions, 0, -1)

    preds_model = get_logits(model, data)
    
    # 2. Definir los grupos de índices a evaluar
    ranges = {'Global': np.arange(total_samples)}
    
    for min_r, max_r in cuts:
        real_min = min_r - 1
        real_max = max_r - 1
        
        mask = (ranks >= real_min) & (ranks <= real_max)
        ranges[f'[{min_r}, {max_r}]'] = np.where(mask)[0]

    results = []

    # 3. Iterar sobre cada rango
    for range_name, indices in ranges.items():
        
        # Calcular el porcentaje que representa este subconjunto
        data_percent = (len(indices) / total_samples) * 100
        
        if len(indices) == 0:
            for metric in metrics:
                results.append({
                    'Range': range_name, 
                    'Data (%)': data_percent,  # Añadimos la columna
                    'Metric': metric.name, 
                    'model': np.nan, 
                    'vcs': np.nan
                })
            continue
            
        y_sub = y[indices]
        preds_model_sub = preds_model[indices]
        preds_vcs_sub = preds_vcs[indices]
        
        y_tensor = torch.tensor(y_sub)
        preds_m_tensor = torch.tensor(preds_model_sub).float()
        preds_v_tensor = torch.tensor(preds_vcs_sub).float()

        # Calcular métricas
        for metric in metrics:
            model_eval = metric.calc(preds_m_tensor, y_tensor)
            vcs_eval = metric.calc(preds_v_tensor, y_tensor)

            results.append({
                'Range': range_name,
                'Data (%)': data_percent,  # Añadimos la columna
                'Metric': metric.name,
                'model': model_eval,
                'vcs': vcs_eval
            })

    # 4. Construcción del DataFrame final
    df = pd.DataFrame(results)
    
    # Redondeamos las métricas a 4 decimales y el porcentaje a 2
    df[['model', 'vcs']] = df[['model', 'vcs']].astype(float).round(4)
    df['Data (%)'] = df['Data (%)'].round(2)

    return df

def data_summary(filename, num_actions):
    data = load_dataset(filename)

    data._open_file()
    try:
        ranks = data.file['ranks'][:]
    finally:
        data.close()

    return ranks

def get_preds(model: Transformer, data):
    model.eval()
    loader = DataLoader(data, batch_size=1) 

    preds = []
    with torch.no_grad():
        for batch in loader:
            *inputs, y_batch = [i for i in batch]
            logits = model(*inputs)
            preds.append(logits.squeeze())
    
    return np.array(preds)

def get_logits(model: Transformer, data):
    model.eval()
    loader = DataLoader(data, batch_size=1) 
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    all_logits = []
    with torch.no_grad():
        for inputs, y_batch in loader:
            inputs = [i for i in inputs]
            logits = model(*inputs)
            all_logits.append(logits.squeeze())
    
    return np.array(all_logits)

def plot_ranking_frequencies(counters_dict):
    """
    Superpone uno o más contadores de frecuencia de ranking.
    
    Args:
        counters_dict (dict): { 'Nombre Serie': [lista_frecuencias], ... }
    """
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Colores consistentes para las comparaciones
    colors = plt.cm.tab10(np.linspace(0, 1, len(counters_dict)))

    for (label, counter), color in zip(counters_dict.items(), colors):
        # 1. Preparar ejes
        positions = np.arange(1, len(counter) + 1)
        
        # 2. Calcular frecuencia acumulada porcentual
        cum_freq = np.cumsum(counter)
        total = cum_freq[-1]
        cum_percent = (cum_freq / total) * 100

        # 3. Graficar
        plt.plot(positions, cum_percent, label=f"{label} (Top-1: {cum_percent[0]:.1f}%)", 
                 linewidth=2, color=color, marker='o', markersize=3, markevery=5)

    # Configuración de estilo
    plt.xlabel("Top-K", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    
    # Límites y rejilla
    plt.ylim(0, 105)
    plt.xlim(0, max([len(c) for c in counters_dict.values()]) + 2)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Leyenda para identificar cada curva
    plt.legend(loc="lower right", frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.show()

def plot_single_ranking_accuracy(counter, label="Modelo"):
    """
    Grafica la precisión acumulada (Top-K Accuracy) para un contador único
    con anotaciones específicas en K=1 y K=16.
    """
    plt.figure(figsize=(10, 6), dpi=100)
    
    # 1. Preparar datos
    positions = np.arange(1, len(counter) + 1)
    cum_freq = np.cumsum(counter)
    total = cum_freq[-1]
    cum_percent = (cum_freq / total) * 100

    # 2. Graficar curva principal
    plt.plot(positions, cum_percent, label=f"{label}", 
             linewidth=2.5, color='#1f77b4', marker='o', 
             markersize=4, markevery=[0, 15]) # Marcadores solo en 1 y 16

    # 3. Anotaciones para K=1 y K=16
    for k_val in [1, 16]:
        idx = k_val - 1
        val = cum_percent[idx]
        
        # Dibujar punto destacado
        plt.scatter(k_val, val, color='red', zorder=5)
        
        # Añadir etiqueta con flecha (anotación)
        plt.annotate(
            f'Top-{k_val}: {val:.1f}%',
            xy=(k_val, val),
            xytext=(k_val + 4, val - 8),
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black')
        )

    # Configuración de estilo
    plt.xlabel("Top-K (Posición)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    
    # Límites y rejilla
    plt.ylim(0, 105)
    plt.xlim(0, len(counter) + 5)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Líneas de referencia opcionales para los ejes en K=1 y K=16
    for k in [1, 16]:
        plt.axvline(x=k, color='red', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('fig.png')
    plt.show()

def get_frequency_dataframe(counter):
    result = [(i, counter[i - 1]) for i in range(1, len(counter) + 1)]
    df = pd.DataFrame(result, columns=["Posición", "Frecuencia"])

    total = df["Frecuencia"].sum()
    df["Frecuencia (%)"] = df["Frecuencia"] / total * 100
    df["Frecuencia acumulada"] = np.cumsum(df["Frecuencia"])
    df["Frecuencia acumulada (%)"] = df["Frecuencia acumulada"] / total * 100

    # Orden final de columnas
    df = df[[
        "Posición",
        "Frecuencia (%)",
        "Frecuencia acumulada (%)"
    ]]

    return df

def calculate_mrr_from_frequencies(arr_counter):
    total_samples = sum(arr_counter)
    cumulative_reciprocal_sum = 0.0

    for i, count in enumerate(arr_counter):
        rank = i + 1  # La posición real (1, 2, 3...)
        cumulative_reciprocal_sum += (1.0 / rank) * count

    mrr = cumulative_reciprocal_sum / total_samples
    return mrr

def results_summary(counter):
    df = get_frequency_dataframe(counter)
    df = df[(df["Posición"] == 1) | (df["Posición"] == 8) | (df["Posición"] == 16)]
    df = df[["Posición", "Frecuencia acumulada (%)"]]
    mrr = calculate_mrr_from_frequencies(counter)
    hmr = mrr ** -1
    display(round(df, 2))
    print("MRR:", f'{mrr:.3f}')
    print("HMR:", f'{hmr:.3f}')
    print("N:", sum(counter))