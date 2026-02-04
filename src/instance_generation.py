import random
from CLP import BoxType, CLPInstance
import os
from settings import INSTANCE_FOLDER


def instance(n_types=10, seed=42):
    """
    Genera una sola instancia del CLP como objeto CLPInstance.
    """
    random.seed(seed)

    # Dimensiones del contenedor (fijas en este caso)
    l, w, h = 587, 233, 220

    # Límites de dimensiones de las cajas
    alpha = [30, 25, 20]   # mínimo largo, ancho, alto
    beta  = [120, 100, 80] # máximo largo, ancho, alto
    L = 2  # constante de estabilidad

    # Volumen del contenedor
    tc = l * w * h

    dimension_box = []
    cantidad_box_type = []
    volumen_box_type = []
    orientacion_box = []

    # Genera tipos de cajas
    for i in range(n_types):
        # Dimensiones aleatorias dentro de los rangos
        r_j = [random.randint(alpha[j], beta[j]) for j in range(3)]
        aux_dim = [alpha[j] + (r_j[j] % (beta[j] - alpha[j] + 1)) for j in range(3)]
        dimension_box.append(aux_dim)

        # Inicializa cantidad
        cantidad_box_type.append(1)

        # Volumen de la caja
        volumen_box_type.append(aux_dim[0] * aux_dim[1] * aux_dim[2])

        # Orientación factible según la constante L
        min_dim = min(aux_dim)
        aux_orient = [1 if aux_dim[j] / min_dim < L else 0 for j in range(3)]
        orientacion_box.append(aux_orient)

    # Rellena hasta que no quepa más
    volumen_cargo = 0
    while True:
        volumen_cargo = sum(cantidad_box_type[i] * volumen_box_type[i] for i in range(n_types))
        aux = random.randint(0, n_types - 1)
        v_k = volumen_box_type[aux]
        if tc > volumen_cargo + v_k:
            cantidad_box_type[aux] += 1
        else:
            break

    # Construye lista de BoxType
    boxes = [
        BoxType(i + 1, dimension_box[i], orientacion_box[i], cantidad_box_type[i])
        for i in range(n_types)
    ]

    return CLPInstance(container=(l, w, h), boxes=boxes)


def generate_instances(filename, n_instances=100, n_types=10, seed=42):
    """
    Genera un archivo con varias instancias CLP en el formato definido.
    """
    # Asegurarse de que la carpeta de salida exista
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)

    # Fijar semilla generadora de números aleatorios
    random.seed(seed)

    with open(INSTANCE_FOLDER / filename, "w") as f:
        # número total de instancias
        f.write(str(n_instances) + "\n")

        for inst_id in range(1, n_instances + 1):
            inst = instance(n_types=n_types, seed=random.random())

            # encabezado de la instancia
            f.write(f"{inst_id}\n")
            l, w, h = inst.container
            f.write(f"{l} {w} {h}\n")
            f.write(f"{inst.n_types}\n")

            # tipos de cajas
            for i, box in enumerate(inst.boxes):
                Lc, Wc, Hc = box.dims
                rotX, rotY, rotZ = box.rots
                qty = box.qty

                if inst_id == n_instances and i == len(inst.boxes) - 1:
                    f.write(f"{box.id} {Lc} {rotX} {Wc} {rotY} {Hc} {rotZ} {qty}")
                else:
                    f.write(f"{box.id} {Lc} {rotX} {Wc} {rotY} {Hc} {rotZ} {qty}\n")

                    
def read_instances(filename) -> list[CLPInstance]:
    """
    Lee un archivo de instancias CLP y devuelve una lista de objetos CLPInstance.
    """
    instances = []
    with open(filename, "r") as f:
        # número total de instancias
        n_instances = int(f.readline().strip())

        for _ in range(n_instances):
            inst_id = int(f.readline().split()[0])  # id de la instancia

            # dimensiones del contenedor
            l, w, h = map(int, f.readline().split())

            # número de tipos de cajas
            n_types = int(f.readline().strip())

            boxes = []
            for _ in range(n_types):
                parts = f.readline().split()
                box_id = int(parts[0])
                Lc = int(parts[1])
                rotX = int(parts[2])
                Wc = int(parts[3])
                rotY = int(parts[4])
                Hc = int(parts[5])
                rotZ = int(parts[6])
                qty = int(parts[7])

                boxes.append(BoxType(box_id, (Lc, Wc, Hc), (rotX, rotY, rotZ), qty))

            instances.append(CLPInstance(container=(l, w, h), boxes=boxes))

    return instances