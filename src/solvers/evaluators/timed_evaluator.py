from solvers.evaluators.utils import print_progress_table, stats_to_df

def timed_eval(solver_list, instance_file, num_instances, min_fr, limit_time, start=0):
    solver_stats = {s.name: {'Vol': []} for s in solver_list}
    instances = list(range(start, start + num_instances))

    for idx, i in enumerate(instances):
        for solver in solver_list:
            vol = solver.solve(instance_file, i, min_fr, limit_time)
            solver_stats[solver.name]['Vol'].append(vol)
        
        print_progress_table(idx + 1, solver_stats)
        
    return stats_to_df(solver_stats, instances)