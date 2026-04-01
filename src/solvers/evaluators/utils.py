from IPython.display import clear_output
import pandas as pd

def print_progress_table(current_inst: int, solver_data: dict):
    clear_output(wait=True)
    solvers = list(solver_data.keys())
    if not solvers: return
    
    # Detectamos las métricas (ej: 'Vol', 'Time')
    metrics = list(solver_data[solvers[0]].keys())
    header = f"{'Solver':<15} | {'Instancia':<10}"
    for m in metrics:
        header += f" | {('Avg ' + m):<12}"
    
    print(header)
    print("-" * len(header))
    
    for name, data in solver_data.items():
        row = f"{name:<15} | {current_inst:<10}"
        for m in metrics:
            # Calculamos el promedio de la lista de datos actual
            avg_val = sum(data[m]) / current_inst
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