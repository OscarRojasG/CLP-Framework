import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

def report_metric(metric_name, phase, filepath='experiment_logs.json'):
    """
    Retorna un DataFrame con la evolución de una métrica para cada dataset 
    en una fase específica, incluyendo una columna 'ponderado' con el promedio.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    phase_key = f"phase_{phase}"
    if phase_key not in data:
        print(f"Error: La fase {phase} no existe en el archivo.")
        return None

    # 1. Extraemos la configuración de pesos y datasets
    config = data.get('config', {})
    datasets = config.get('datasets', [])
    
    # NOTA: Aquí asumo que usas 'loss_weights' también para las métricas. 
    # Si en tu JSON tienes algo como 'metric_weights', cámbialo en la línea de abajo.
    weights = config.get('loss_weights', [])
    
    # Creamos el mapa de pesos por dataset
    weight_map = dict(zip(datasets, weights))

    records = []
    
    # Iteramos sobre las épocas de la fase
    for epoch_key, epoch_data in data[phase_key].items():
        epoch_num = int(epoch_key.split('_')[1])
        
        # Obtenemos las métricas de validación
        val_metrics = epoch_data['val']['metrics']
        
        row = {'epoch': epoch_num}
        
        # Agregamos cada dataset como una columna
        for ds_name, metrics in val_metrics.items():
            row[ds_name] = metrics.get(metric_name, None)
            
        records.append(row)

    # Creamos el DF, indexamos por época y ordenamos
    df = pd.DataFrame(records).set_index('epoch').sort_index()

    # --- CÁLCULO DEL PROMEDIO PONDERADO ---
    # Identificamos qué columnas son datasets válidos y tienen peso asignado
    valid_cols = [col for col in datasets if col in df.columns]
    
    if valid_cols and weights:
        valid_weights = [weight_map[col] for col in valid_cols]
        # Calculamos el promedio ponderado usando numpy
        df['ponderado'] = np.average(df[valid_cols], axis=1, weights=valid_weights)
    else:
        # Fallback de seguridad por si no se encuentran los pesos en el JSON
        print("Advertencia: No se encontraron pesos válidos, se calculó un promedio simple.")
        df['ponderado'] = df.mean(axis=1)
    
    df.index.name = None
    return df

def report_loss(phase, filepath='experiment_logs.json'):
    """
    Retorna un DataFrame con la evolución del Loss de validación 
    para cada dataset y una columna 'promedio' ponderado adicional.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    phase_key = f"phase_{phase}"
    if phase_key not in data:
        print(f"Error: La fase {phase} no existe en el archivo.")
        return None

    # 1. Extraemos la configuración de pesos y datasets
    config = data.get('config', {})
    datasets = config.get('datasets', [])
    weights = config.get('loss_weights', [])
    
    # Creamos un diccionario para mapear { 'nombre_dataset': peso }
    # Ej: {'easy.data': 0.3, 'medium.data': 0.6, 'hard.data': 0.1}
    weight_map = dict(zip(datasets, weights))

    records = []
    
    # Iteramos sobre las épocas de la fase
    for epoch_key, epoch_data in data[phase_key].items():
        epoch_num = int(epoch_key.split('_')[1])
        
        # Obtenemos los losses directamente
        losses = epoch_data.get('val', {}).get('losses', {})
        
        row = {'epoch': epoch_num}
        row.update(losses) 
        records.append(row)

    # Creamos el DataFrame
    df = pd.DataFrame(records).set_index('epoch').sort_index()
    
    # --- AQUÍ ESTÁ EL CAMBIO ---
    # Filtramos para usar solo las columnas (datasets) que existen tanto en el DataFrame 
    # como en la configuración, evitando fallos por datasets extra o faltantes.
    valid_cols = [col for col in datasets if col in df.columns]
    valid_weights = [weight_map[col] for col in valid_cols]
    
    # Calculamos el promedio ponderado a través de las columnas (axis=1)
    # Utilizamos np.average que soporta el argumento 'weights' de forma nativa
    df['ponderado'] = np.average(df[valid_cols], axis=1, weights=valid_weights)
    # ---------------------------
    
    df.index.name = None
    return df

def plot_curves(df, y_axis_name="Loss", title=None):
    """
    Recibe el DataFrame con los losses y grafica una curva para 
    cada dataset junto con la curva del promedio ponderado.
    """
    if df is None or df.empty:
        print("Error: El DataFrame está vacío o no es válido.")
        return

    # Configuramos el tamaño del gráfico
    plt.figure(figsize=(10, 6))

    # Iteramos sobre cada columna del DataFrame
    for column in df.columns:
        if column == 'promedio':
            # Destacamos la curva del promedio ponderado
            plt.plot(df.index, df[column], label='Promedio Ponderado', 
                     color='black', linewidth=3, linestyle='--')
        else:
            # Graficamos los datasets individuales de forma estándar
            plt.plot(df.index, df[column], label=f'{column}', 
                     linewidth=1.5, alpha=0.8)

    # Añadimos etiquetas, título y leyenda
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Época", fontsize=12)
    plt.ylabel(y_axis_name, fontsize=12)
    
    # Colocamos la leyenda
    plt.legend(fontsize=11)
    
    # Añadimos una cuadrícula para facilitar la lectura
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Ajustamos los márgenes
    plt.tight_layout()
    
    # Mostramos el gráfico
    plt.show()

def compare_train_val_loss(phase, filepath='experiment_logs.json'):
    """
    Retorna un DataFrame que compara el Loss ponderado de entrenamiento (train)
    contra el Loss ponderado de validación (val) para cada época.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    phase_key = f"phase_{phase}"
    if phase_key not in data:
        print(f"Error: La fase {phase} no existe en el archivo.")
        return None

    # Extraemos la configuración oficial de datasets y pesos
    config = data.get('config', {})
    datasets = config.get('datasets', [])
    weights = config.get('loss_weights', [])
    weight_map = dict(zip(datasets, weights))

    records = []

    # Iteramos sobre las épocas de la fase
    for epoch_key, epoch_data in data[phase_key].items():
        epoch_num = int(epoch_key.split('_')[1])
        
        # Extraemos los diccionarios de pérdidas
        train_losses = epoch_data.get('train', {}).get('losses', {})
        val_losses = epoch_data.get('val', {}).get('losses', {})
        
        # --- CÁLCULO TRAIN PONDERADO ---
        # Filtramos estrictamente por los datasets de la configuración
        valid_train_cols = [col for col in datasets if col in train_losses]
        if valid_train_cols:
            train_vals = [train_losses[col] for col in valid_train_cols]
            train_w = [weight_map[col] for col in valid_train_cols]
            weighted_train = np.average(train_vals, weights=train_w)
        else:
            weighted_train = np.nan

        # --- CÁLCULO VAL PONDERADO ---
        # Filtramos estrictamente por los datasets de la configuración
        valid_val_cols = [col for col in datasets if col in val_losses]
        if valid_val_cols:
            val_vals = [val_losses[col] for col in valid_val_cols]
            val_w = [weight_map[col] for col in valid_val_cols]
            weighted_val = np.average(val_vals, weights=val_w)
        else:
            weighted_val = np.nan

        # Guardamos el resumen de la época
        records.append({
            'epoch': epoch_num,
            'train_loss': weighted_train,
            'val_loss': weighted_val
        })

    # Creamos el DataFrame comparativo
    df = pd.DataFrame(records).set_index('epoch').sort_index()
    df.index.name = None
    
    return df

def plot_train_val_comparison(df, y_axis_name="Loss",title=None):
    """
    Recibe un DataFrame con columnas 'train_loss' y 'val_loss' 
    y genera un gráfico lineal comparativo.
    """
    # Verificación de seguridad
    if df is None or df.empty:
        print("Error: El DataFrame está vacío o no es válido.")
        return

    # Configuramos el tamaño de la figura
    plt.figure(figsize=(6, 6))

    # Graficamos la curva de entrenamiento (Train) si existe en el DataFrame
    if 'train_loss' in df.columns:
        plt.plot(df.index, df['train_loss'], label='Train Loss', 
                 color='#1f77b4', linewidth=2, markersize=4)
    
    # Graficamos la curva de validación (Val) si existe en el DataFrame
    if 'val_loss' in df.columns:
        plt.plot(df.index, df['val_loss'], label='Val Loss', 
                 color='#d62728', linewidth=2, markersize=4)

    # Configuramos los textos y estética del gráfico
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Época", fontsize=12)
    plt.ylabel(y_axis_name, fontsize=12)
    
    # Añadimos la leyenda y la grilla
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Ajustamos los márgenes para que no se corte ningún texto
    plt.tight_layout()
    
    # Renderizamos el gráfico
    plt.show()