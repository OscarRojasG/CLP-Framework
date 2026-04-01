from solvers.evaluators.utils import print_progress_table, stats_to_df

def dse_eval(solver_list, instance_file, num_instances, max_w, min_fr):
    solver_stats = {s.name: {'Vol': [], 'Time': []} for s in solver_list}
    instances = list(range(num_instances))

    for i in instances:
        for solver in solver_list:
            vol, time = solver.solve(instance_file, i, min_fr, max_w)
            solver_stats[solver.name]['Vol'].append(vol)
            solver_stats[solver.name]['Time'].append(time)
            
        print_progress_table(i + 1, solver_stats)
        
    return stats_to_df(solver_stats, instances)