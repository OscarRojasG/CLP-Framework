from IPython.display import clear_output
import pandas as pd

def print_progress_table(solver_data: dict, total_inst: int, is_final: bool = False):
    # El truco: si es la última ejecución, wait=False borra de inmediato 
    # y rompe el ciclo de reemplazo para que el DataFrame no la borre.
    clear_output(wait=not is_final)
    
    solvers = list(solver_data.keys())
    if not solvers: return
    
    metrics = list(solver_data[solvers[0]].keys())
    
    header = f"{'Solver':<15} | {'Progreso':<10}"
    for m in metrics:
        header += f" | {('Avg ' + m):<12}"
    
    print(header)
    print("-" * len(header))
    
    for name, data in solver_data.items():
        # Contamos el total de entradas registradas para el progreso
        total_intentos = len(data[metrics[0]]) if metrics else 0
        row = f"{str(name):<15} | {f'{total_intentos}/{total_inst}':<10}"
        
        for m in metrics:
            # 1. Extraemos solo los valores válidos (no None)
            valores_validos = [val for inst_id, val in data[m] if val is not None]
            
            # 2. Calculamos el promedio sobre la cantidad de valores encontrados
            cantidad_valida = len(valores_validos)
            if cantidad_valida > 0:
                avg_val = sum(valores_validos) / cantidad_valida
            else:
                avg_val = 0.0
                
            row += f" | {avg_val:<12.2f}"
            
        print(row)

def stats_to_df(solver_stats: dict, instances: list):
    """Convierte el diccionario anidado en un DataFrame plano."""
    final_data = {'instance': instances}
    
    for s_name, metrics in solver_stats.items():
        for m_name, values in metrics.items():
            col_name = f"{m_name} {s_name}"
            final_data[col_name] = values
            
    return pd.DataFrame(final_data)