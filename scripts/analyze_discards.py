"""
Analiza carpetas con archivos .out (salida de run_instances_parallel) y simula
los descartes de generate_data y el particionado por rangos de generate_datasets,
sin construir ningún dataset.

Uso: ajustar los parámetros de configuración y ejecutar
    python scripts/analyze_discards.py
"""
import os
from concurrent.futures import ProcessPoolExecutor
from settings import OUTPUT_FOLDER

# ─── Parámetros de configuración ───────────────────────────────────────────────
FOLDERS = ["E15-42", "E15-43", "E15-44"]  # Subcarpetas de OUTPUT_FOLDER a analizar
MIN_BLOCKS = 100
MIN_ACTIONS = 4
MAX_ACTIONS = 64
CUTS = [1, 1, 8, 64]
MAX_WORKERS = None
# ───────────────────────────────────────────────────────────────────────────────

NOT_FOUND = 999999  # Mismo valor que get_rank cuando el bloque no está en las acciones


def read_steps(filepath):
    """
    Versión ligera de data.generation.read_output: recorre el archivo con la
    misma estructura, pero solo extrae lo necesario para los filtros.
    Retorna (n_blocks, [(n_actions, rank), ...]).
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    it = iter(lines)
    n_blocks = 0

    for line in it:
        if line == "BLOCKS":
            for l in it:
                if l == "SOLVE STEPS":
                    break
                n_blocks += 1
            break

    steps = []
    for line in it:
        if line != "Actions":
            continue

        action_ids = []
        for l in it:
            if l == "Placed":
                break
            action_ids.append(int(float(l.split(maxsplit=1)[0])))

        for l in it:
            if l == "Space":
                break

        next(it)  # Datos de Space
        next(it)  # "Selected Block"
        selected_block = int(next(it))

        next(it)  # "Greedy"
        for l in it:
            if l == "Volume":
                break
        next(it)  # Volumen actual

        try:
            rank = action_ids.index(selected_block)
        except ValueError:
            rank = NOT_FOUND
        steps.append((len(action_ids), rank))

    return n_blocks, steps


def analyze_file(filepath, min_blocks, min_actions, max_actions, cuts_ranges):
    n_blocks, steps = read_steps(filepath)

    stats = {
        "instances": 1,
        "total": len(steps),
        "min_blocks": 0,
        "rank": 0,
        "min_actions": 0,
        "union": 0,
        "valid": 0,
        "cuts": [0] * len(cuts_ranges),
        "out_of_cuts": 0,
    }

    few_blocks = n_blocks < min_blocks
    for n_actions, rank in steps:
        c_blocks = few_blocks
        c_rank = rank >= max_actions
        c_actions = n_actions < min_actions

        stats["min_blocks"] += c_blocks
        stats["rank"] += c_rank
        stats["min_actions"] += c_actions

        if c_blocks or c_rank or c_actions:
            stats["union"] += 1
            continue

        stats["valid"] += 1
        # Mismo criterio que split_data: start_cut - 1 <= rank <= end_cut - 1
        for j, (start_cut, end_cut) in enumerate(cuts_ranges):
            if start_cut - 1 <= rank <= end_cut - 1:
                stats["cuts"][j] += 1
                break
        else:
            stats["out_of_cuts"] += 1

    return stats


def merge(a, b):
    for k, v in b.items():
        if k == "cuts":
            a[k] = [x + y for x, y in zip(a[k], v)]
        else:
            a[k] += v
    return a


def empty_stats(n_cuts):
    return {"instances": 0, "total": 0, "min_blocks": 0, "rank": 0, "min_actions": 0, "union": 0,
            "valid": 0, "cuts": [0] * n_cuts, "out_of_cuts": 0}


def build_cut_ranges(cuts):
    # Mismo criterio que generate_datasets
    return [(cuts[j] if j == 0 else cuts[j] + 1, cuts[j + 1]) for j in range(len(cuts) - 1)]


def pct(x, total):
    return f"{100 * x / total:6.2f}%" if total else "   -   "


def print_report(name, s, cuts_ranges):
    total, valid = s["total"], s["valid"]

    def row(label, value, ref, indent=2):
        print(f"{' ' * indent}{label:<48}{value:>10}  {pct(value, ref)}")

    print(f"\n═══ {name} ═══")
    print(f"  {'Instancias procesadas':<48}{s['instances']:>10}")
    print(f"  {'Datos totales iniciales':<48}{total:>10}")
    row(f"Descartados por min_blocks (< {MIN_BLOCKS})", s["min_blocks"], total)
    row(f"Descartados por rank >= max_actions ({MAX_ACTIONS})", s["rank"], total)
    row(f"Descartados por min_actions (< {MIN_ACTIONS})", s["min_actions"], total)
    row("Descartados (unión de las 3 condiciones)", s["union"], total)
    row("Datos válidos", valid, total)

    print(f"  Distribución de válidos por rango del ranking (cuts={CUTS}):")
    for (start, end), count in zip(cuts_ranges, s["cuts"]):
        row(f"[{start}, {end}]", count, valid, indent=4)
    if s["out_of_cuts"]:
        row("fuera de los cortes", s["out_of_cuts"], valid, indent=4)


def main():
    cuts_ranges = build_cut_ranges(CUTS)

    groups = {}
    for folder in FOLDERS:
        folder_path = OUTPUT_FOLDER / folder
        if not folder_path.is_dir():
            print(f"Carpeta no encontrada: {folder_path}")
            continue
        outs = sorted(f for f in os.listdir(folder_path) if f.endswith(".out"))
        if not outs:
            print(f"Sin archivos .out: {folder_path}")
            continue
        groups[folder] = [str(folder_path / f) for f in outs]

    if not groups:
        return

    total_stats = empty_stats(len(cuts_ranges))
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for name, filepaths in groups.items():
            n = len(filepaths)
            results = executor.map(
                analyze_file, filepaths,
                [MIN_BLOCKS] * n, [MIN_ACTIONS] * n,
                [MAX_ACTIONS] * n, [cuts_ranges] * n,
                chunksize=8,
            )
            folder_stats = empty_stats(len(cuts_ranges))
            for r in results:
                merge(folder_stats, r)
            print_report(f"{name} ({n} archivos)", folder_stats, cuts_ranges)
            merge(total_stats, folder_stats)

    if len(groups) > 1:
        print_report(f"TOTAL ({len(groups)} carpetas)", total_stats, cuts_ranges)


if __name__ == "__main__":
    main()
