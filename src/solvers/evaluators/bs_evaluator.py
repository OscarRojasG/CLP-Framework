from solvers.evaluators.utils import print_progress_table, stats_to_df
from training.training import load_model
import concurrent.futures
from collections import defaultdict
from solvers.env_solver import EnvSolver
import torch
import os
from collections import defaultdict
import pandas as pd
from settings import EXPERIMENTS_FOLDER

def bs_eval(solver_list, instance_file, num_instances):
    # Usamos defaultdict para mayor seguridad
    solver_stats = defaultdict(lambda: {'Vol': [], 'Time': []})
    instances = list(range(num_instances))

    for i in instances:
        for solver in solver_list:
            try:
                vol, time = solver.solve(instance_file, i)
                # Guardamos explícitamente el índice para evitar desajustes
                solver_stats[solver.name]['Vol'].append((i, vol))
                solver_stats[solver.name]['Time'].append((i, time))
            except Exception as e:
                print(f"Error en {solver.name} en instancia {i}: {e}")
                # Opcional: rellenar con None o valores por defecto para mantener consistencia
                solver_stats[solver.name]['Vol'].append((i, None))
                solver_stats[solver.name]['Time'].append((i, None))
            
        print_progress_table(solver_stats, num_instances, is_final=(i == num_instances - 1)) # Asegúrate que tu función maneje dicts y no solo listas
        
    # Limpieza: convertir a dict y ordenar (igual que en fast_eval)
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
    
    model = load_model(model_cls, model_name)
    
    adapter_cls, *adapter_args = adapter_config
    input_adapter = adapter_cls(*adapter_args)
    
    for solver_cls, *args in solvers_config:
        if issubclass(solver_cls, EnvSolver):
            # Inyectamos el modelo y el adapter, y luego desempaquetamos el resto
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

def fast_eval(solvers_config, instance_file, num_instances, model_cls, model_name, adapter_config, num_workers=4, backup_csv="respaldo_evaluacion.csv"):
    
    # (Opcional) Limpiar el archivo de respaldo si existe de una ejecución anterior
    backup_csv = EXPERIMENTS_FOLDER / backup_csv
    if os.path.exists(backup_csv):
        os.remove(backup_csv)
        
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
                # Ya no abortamos. Imprimimos el error y registramos valores nulos.
                print(f"[Advertencia] Solver: {solver_name} | Instancia: {i} | Falló con error: {error}")
                
                solver_stats[solver_name]['Vol'].append((i, None))
                solver_stats[solver_name]['Time'].append((i, None))
            else:
                solver_stats[solver_name]['Vol'].append((i, vol))
                solver_stats[solver_name]['Time'].append((i, time))
            
            # --- NUEVO: Guardar respaldo en CSV progresivamente ---
            df_row = pd.DataFrame([{
                'Solver': solver_name,
                'Instancia': i,
                'Vol': vol if not error else None,
                'Time': time if not error else None,
                'Error': str(error) if error else None
            }])
            
            # header=True solo si el archivo aún no se ha creado
            es_nuevo = not os.path.exists(backup_csv)
            df_row.to_csv(backup_csv, mode='a', header=es_nuevo, index=False)
            # ------------------------------------------------------

            es_el_final = (count == len(tareas))
            print_progress_table(solver_stats, num_instances, is_final=es_el_final)

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
    return stats_to_df(solver_stats, instances)