from ..env import Environment, State, Action
from ..training import get_predictions
import torch
from torch import optim
from torch.distributions import Categorical
from ..models.base.encoder_decoder_pe import EncoderDecoderPEModel

env = Environment()

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
        output = output.squeeze(0)
        dist = Categorical(output)
        action_idx = dist.sample() 
        prob = output[action_idx]
        selected_action = valid_actions[action_idx]

        # Aplicar la acción
        reward = env.state_transition(state, selected_action)


        # Acumular variables
        rewards.append(reward)
        probs.append(prob)

    vol_ratio = state.get_volume_ratio()

    state.close()
    return probs, rewards, vol_ratio

# Generamos datos RL por batch
def generate_rl_data_batch(model, instance_file, instance_number, batch_size):
    all_probs = []       # Lista de listas de probabilidades
    all_rewards = []     # Lista de listas de recompensas
    all_vol_ratios = []  # Lista de ratios de volumen

    for _ in range(batch_size):
        probs, rewards, vol_ratio = generate_rl_data(model, instance_file, instance_number)
        all_probs.append(probs)      # Dejamos cada secuencia como lista
        all_rewards.append(rewards)
        all_vol_ratios.append(vol_ratio)

    return all_probs, all_rewards, all_vol_ratios

# Calcular retorno descontado
def compute_discounted_returns(rewards, gamma):
    discounted_returns = []
    accumulated_return = 0
    for reward in reversed(rewards):
        accumulated_return = reward + gamma * accumulated_return
        discounted_returns.insert(0, accumulated_return)

    discounted_returns = torch.tensor(discounted_returns)
    return discounted_returns


# Actualizar parámetros (con datos de batch)
def update_params(optimizer, all_probs, all_discounted_rewards):
    losses = []

    for probs, rewards in zip(all_probs, all_discounted_rewards):
        log_probs = torch.log(torch.stack(probs))
        rewards = torch.tensor(rewards, dtype=torch.float32)
        losses.append(-(log_probs * rewards).mean())

    # Promediar pérdidas de todos los episodios en el batch
    loss = torch.stack(losses).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

# Entrenamiento con REINFORCE por batch
def reinforce(model, instance_file, instance_number, eps=10, lr=1e-4, gamma=0.99, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    vol_arr = []
    loss_arr = []

    for i in range(eps):
        # Entrenar y actualizar parámetros por episodios (en batch)
        all_probs, all_rewards, all_vol_ratios = generate_rl_data_batch(model, instance_file, instance_number, batch_size)
        
        # Calcular retornos descontados por batch
        all_discounted_rewards = [compute_discounted_returns(rewards, gamma) for rewards in all_rewards]

        # Actualizar parámetros
        loss = update_params(optimizer, all_probs, all_discounted_rewards)

        vol_arr.append(torch.tensor(all_vol_ratios, dtype=torch.float32).mean().item())
        loss_arr.append(float(loss))
        print(f'Episode {i + 1}/{eps} - Loss: {loss} - Volume Ratio: {vol_arr[-1]}')

    return vol_arr, loss_arr