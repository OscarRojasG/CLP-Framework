import subprocess
import fcntl
import os

state_folder_path = "states/"

class State:
    def __init__(self, process: subprocess.Popen):
        self.process = process  # Referencia al proceso persistente
        self.occupied_volume, self.total_volume = self.update()

    def close(self):
        self.process.stdin.write("-Q\n")
        self.process.stdin.flush()

    def update(self):
        self.process.stdin.write("-V\n")
        self.process.stdin.flush()

        line = self.process.stdout.readline().strip()
        parts = line.split()

        try:
            occupied_volume = int(float(parts[0]))
            total_volume = int(float(parts[1]))
        except Exception:
            raise RuntimeError("Error al actualizar datos del estado.")
        
        return occupied_volume, total_volume
    
    def get_total_volume(self):
        return self.total_volume
    
    def get_occupied_volume(self):
        return self.occupied_volume
    
    def get_volume_ratio(self):
        if self.total_volume == 0:
            return 0.0
        return self.occupied_volume / self.total_volume
    
class Action:
    def __init__(self, block_id, action_vec):
        self.block_id = block_id
        self.action_vec = action_vec

class Environment:
    @staticmethod
    def initial_state(instance_file, instance_number, w: int) -> State:
        """
        Inicia el proceso persistente de BSG_ENV y configura el estado inicial.
        """
        # Crear proceso persistente
        process = subprocess.Popen(
            [
                "./../Metasolver/BSG_ENV",
                f"instances/{instance_file}",
                "-i", str(instance_number),
                "-w", str(w)
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        process.stdout.readline() # Leer línea inicial
        return State(process)

    @staticmethod
    def get_valid_actions(state: State) -> list[Action]:
        """
        Solicita al proceso las acciones válidas a partir del estado actual.
        Formato esperado por línea (separado por espacios):
            block_id val1 val2 val3 ... valX
        La salida siempre termina con una línea "END".
        """
        process = state.process

        process.stdin.write("-A\n")
        process.stdin.flush()

        actions = []
        end_found = False

        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("Salida inesperada: no se encontró 'END'.")

            line = line.strip()

            if line == "END":
                end_found = True
                break

            parts = line.split()
            try:
                block_id = int(parts[0])
                action_vec = [float(v) for v in parts[1:5]]
            except Exception:
                raise RuntimeError("Formato de salida inválido en la respuesta de BSG_ENV.")

            actions.append(Action(block_id=block_id, action_vec=action_vec))

        if not end_found:
            raise RuntimeError("Salida inesperada: no se encontró 'END'.")

        if not actions:
            return []

        return actions

    @staticmethod
    def state_transition(state: State, action: 'Action'):
        """
        Envía una acción al proceso para realizar la transición de estado.
        """
        process = state.process
        cmd = f"-T {action.block_id}\n"
        process.stdin.write(cmd)
        process.stdin.flush()

        return State(process)