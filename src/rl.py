from .env import Environment, State, Action
from .training import get_predictions
import torch
from torch import optim
from .models.base.encoder_decoder_pe import EncoderDecoderPEModel

env = Environment

# Generamos datos RL
def generate_rl_data(model, instance_file, instance_number):
    probs = [] # Secuencia de probabilidades
    rewards = [] # Secuencia de recompensas

    # Generar estado inicial
    state = env.initial_state(instance_file, instance_number, w=8)

    # Generar secuencia de acciones
    while True:
        try:
            add_block_index = False
            if isinstance(model, EncoderDecoderPEModel):
                add_block_index = True
            valid_actions = env.get_valid_actions(state, add_block_index)
        except Exception as e:
            print("Error obteniendo acciones válidas:", e)
            state.close()
            raise

        if len(valid_actions) == 0: break # Estado completado

        # Convertir acciones a vectores
        action_vec = [action.action_vec for action in valid_actions]
        X_tgt = torch.tensor([action_vec], dtype=torch.float32)

        # Datos
        X_src = torch.tensor([state.get_blocks()], dtype=torch.float32)
        placed = torch.tensor([state.get_placed()], dtype=torch.float32)
        coords = torch.tensor([state.get_coords()], dtype=torch.float32)

        # Predecir la mejor acción
        output = get_predictions(model, X_src, X_tgt, placed, coords, apply_softmax=True)
        action_idx = output.argmax()
        prob = output[0, action_idx]
        selected_action = valid_actions[action_idx]

        # Aplicar la acción
        reward = env.state_transition(state, selected_action)

        # Acumular variables
        rewards.append(reward)
        probs.append(prob)

    vol_ratio = state.get_volume_ratio()

    state.close()
    return probs, rewards, vol_ratio

# Calcular retorno descontado
def compute_discounted_returns(rewards, gamma):
    discounted_returns = []
    accumulated_return = 0
    for reward in reversed(rewards):
        accumulated_return = reward + gamma * accumulated_return
        discounted_returns.insert(0, accumulated_return)

    discounted_returns = torch.tensor(discounted_returns)
    return discounted_returns

# Actualizar parámetros
def update_params(optimizer, probs, rewards):
    log_probs = torch.log(torch.stack(probs))
    rewards = torch.tensor(rewards, dtype=torch.float32)
    loss = -(log_probs * rewards).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

# Entrenamiento con REINFORCE
def reinforce(model, instance_file, instance_number, eps=10, lr=1e-4, gamma=0.99):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    vol_arr = []
    loss_arr = []

    for i in range(eps):
        # Entrenar y actualizar parámetros por episodio
        probs, rewards, vol = generate_rl_data(model, instance_file, instance_number)
        discounted_r = compute_discounted_returns(rewards, gamma)
        loss = update_params(optimizer, probs, discounted_r)

        vol_arr.append(vol)
        loss_arr.append(float(loss))
        print(f'Volume Ratio: {vol}\tLoss: {loss}')

    return vol_arr, loss_arr