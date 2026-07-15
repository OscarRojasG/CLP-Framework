from training.training import load_model
import concurrent.futures
from collections import defaultdict
from solvers.env_solver import EnvSolver
import torch
import os
from collections import defaultdict
import pandas as pd
from settings import EXPERIMENTS_FOLDER

def run_eval(solver_list, instance_file, num_instances, output_csv=None):
    # Si se proporciona un nombre de archivo, preparamos la ruta y limpiamos ejecuciones previas
    if output_csv:
        output_path = EXPERIMENTS_FOLDER / output_csv
        if os.path.exists(output_path):
            os.remove(output_path)

    solver_stats = defaultdict(lambda: {'Vol': [], 'Time': []})
    instances = list(range(num_instances))

    for i in instances:
        for s_idx, solver in enumerate(solver_list):
            is_first_call = (i == 0 and s_idx == 0) 
            
            # Inicializamos variables para manejar el respaldo correctamente
            vol = None
            time = None
            error_msg = None
            
            try:
                vol, time = solver.solve(instance_file, i)
                solver_stats[solver.name]['Vol'].append((i, vol))
                solver_stats[solver.name]['Time'].append((i, time))
                
                print_progress_table(solver.name, i, vol, time, is_first=is_first_call)
                
            except Exception as e:
                error_msg = str(e)
                solver_stats[solver.name]['Vol'].append((i, None))
                solver_stats[solver.name]['Time'].append((i, None))
                
                print_progress_table(solver.name, i, None, None, error=error_msg, is_first=is_first_call)
            
            # Guardar respaldo en CSV progresivamente si se solicitó
            if output_csv:
                df_row = pd.DataFrame([{
                    'Solver': solver.name,
                    'Instancia': i,
                    'Vol': vol,
                    'Time': time,
                    'Error': error_msg
                }])
                
                es_nuevo = not os.path.exists(output_path)
                df_row.to_csv(output_path, mode='a', header=es_nuevo, index=False)
        
    # Limpieza: convertir a dict y ordenar
    final_stats = dict(solver_stats)
    for name in final_stats:
        final_stats[name]['Vol'].sort(key=lambda x: x[0])
        final_stats[name]['Time'].sort(key=lambda x: x[0])
        final_stats[name]['Vol'] = [val for idx, val in final_stats[name]['Vol']]
        final_stats[name]['Time'] = [val for idx, val in final_stats[name]['Time']]
        
    return stats_to_df(final_stats, instances)

global_solvers = []

def init_worker(solvers_config, model_cls, model_name, adapter_config):
    global global_solvers

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    
    for solver_cls, *args in solvers_config:
        if issubclass(solver_cls, EnvSolver):
            model = load_model(model_cls, model_name)
            adapter_cls, *adapter_args = adapter_config
            input_adapter = adapter_cls(*adapter_args)

            solver = solver_cls(model, input_adapter, *args)
        else:
            solver = solver_cls(*args)
            
        global_solvers.append(solver)

def worker_task(solver_name, instance_file, i):
    """
    La tarea atómica que ejecuta el solver.
    """
    try:
        solver = global_solvers[solver_name]
        vol, time = solver.solve(instance_file, i)
        return (solver.name, i, vol, time, None)
    except Exception as e:
        return (solver.name, i, None, None, str(e))

def fast_eval(solvers_config, instance_file, num_instances, model_cls=None, model_name=None, adapter_config=None, num_workers=None, output_csv=None):
    # 1. defaultdict crea la estructura {'Vol': [], 'Time': []} mágicamente la primera vez que ve una clave nueva
    solver_stats = defaultdict(lambda: {'Vol': [], 'Time': []})
    
    # 2. Armamos las tareas usando solo el ÍNDICE de la configuración
    tareas = []
    for i in range(num_instances):
        for idx in range(len(solvers_config)):
            tareas.append((idx, instance_file, i))
            
    print(f"Iniciando evaluación: {len(tareas)} tareas en total...")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(solvers_config, model_cls, model_name, adapter_config)
    ) as executor:
        
        futuros = {executor.submit(worker_task, *tarea): tarea for tarea in tareas}
        
        for count, futuro in enumerate(concurrent.futures.as_completed(futuros), 1):
            solver_name, i, vol, time, error = futuro.result()
            
            if error:
                solver_stats[solver_name]['Vol'].append((i, None))
                solver_stats[solver_name]['Time'].append((i, None))
            else:
                solver_stats[solver_name]['Vol'].append((i, vol))
                solver_stats[solver_name]['Time'].append((i, time))

            # --- NUEVA LLAMADA A LA TABLA ---
            es_el_primero = (count == 1)
            print_progress_table(solver_name, i, vol, time, error=error, is_first=es_el_primero)

    # Convertimos el defaultdict a dict normal para procesarlo
    solver_stats = dict(solver_stats)

    for name in solver_stats:
        # Ordenamos por índice de instancia para que queden alineados
        solver_stats[name]['Vol'].sort(key=lambda x: x[0])
        solver_stats[name]['Time'].sort(key=lambda x: x[0])
        
        # Extraemos solo los valores, descartando el índice
        solver_stats[name]['Vol'] = [val for idx, val in solver_stats[name]['Vol']]
        solver_stats[name]['Time'] = [val for idx, val in solver_stats[name]['Time']]

    instances = list(range(num_instances))
    df_results = stats_to_df(solver_stats, instances)
    
    # Guardar todos los resultados al final si se solicitó
    if output_csv:
        output_path = EXPERIMENTS_FOLDER / output_csv
        df_results.to_csv(output_path, index=False)
        print(f"\nResultados guardados exitosamente en: {output_path}")

    return df_results

def print_progress_table(solver_name: str, instance_id: int, vol: float, time: float, error=None, is_first: bool = False):
    """Imprime una fila de progreso para la instancia recién completada."""
    
    # Imprimir el encabezado solo en la primera llamada
    if is_first:
        header = f"{'Instancia':<10} | {'Solver':<15} | {'Volumen':<12} | {'Tiempo':<12} | {'Estado':<20}"
        print(header)
        print("-" * len(header))
        
    # Formatear valores manejando posibles errores o nulos
    estado = f"Error: {error}" if error else "OK"
    vol_str = f"{vol:.4f}" if vol is not None else "N/A"
    time_str = f"{time:.4f}" if time is not None else "N/A"
    
    # Imprimir la fila de la instancia actual
    print(f"{instance_id:<10} | {solver_name:<15} | {vol_str:<12} | {time_str:<12} | {estado:<20}")

def stats_to_df(solver_stats: dict, instances: list):
    """Convierte el diccionario anidado en un DataFrame plano."""
    final_data = {'instance': instances}
    
    for s_name, metrics in solver_stats.items():
        for m_name, values in metrics.items():
            col_name = f"{m_name} {s_name}"
            final_data[col_name] = values
            
    return pd.DataFrame(final_data)