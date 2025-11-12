from ..env import Environment, State, Action
from ..training import get_predictions
import torch
from torch import optim
from torch.distributions import Categorical
import torch.nn.functional as F
from ..models.base.encoder_decoder_pe import EncoderDecoderPEModel
import numpy as np

env = Environment()

# Generamos datos RL
def generate_rl_data(model, instance_file, instance_number):
    all_probs = [] # Secuencia de probabilidades
    all_rewards = [] # Secuencia de recompensas

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

        output = get_predictions(model, X_src, X_tgt, placed, coords)  # (1, 64)
        output = output.squeeze(0)                                     # -> (64,)
        probs = F.softmax(output, dim=-1)

        dist = Categorical(probs)
        action_idx = dist.sample()

        prob = probs[action_idx]
        selected_action = valid_actions[action_idx]

        # Aplicar la acción
        reward = env.state_transition(state, selected_action)

        # Acumular variables
        all_rewards.append(reward)
        all_probs.append(prob)

    vol_ratio = state.get_volume_ratio()

    state.close()
    return all_probs, all_rewards, vol_ratio

# ===============================================
# 🔹 Calcular retorno descontado
# ===============================================
def compute_discounted_returns(rewards, gamma):
    discounted_returns = []
    accumulated_return = 0
    for reward in reversed(rewards):
        accumulated_return = reward + gamma * accumulated_return
        discounted_returns.insert(0, accumulated_return)
    return torch.tensor(discounted_returns, dtype=torch.float32)


# ===============================================
# 🔹 Actualizar parámetros (con baseline y normalización)
# ===============================================
def update_params(optimizer, batch_log_probs, batch_returns):
    # Concatenar todos los episodios del batch
    log_probs = torch.cat(batch_log_probs)
    returns = torch.cat(batch_returns)

    # Baseline constante (media del episodio)
    #baseline = returns.mean()
    #advantage = returns - baseline
    advantage = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Calcular pérdida
    loss = -(log_probs * advantage).mean()

    # Backprop
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    #torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
    optimizer.step()

    return loss


# ===============================================
# 🔹 Entrenamiento REINFORCE con reducción de varianza
# ===============================================
def reinforce(model, instance_file, instance_number, eps=100, lr=1e-4, gamma=0.99, batch_size=8):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    vol_arr = []
    loss_arr = []

    model.train()

    for i in range(0, eps, batch_size):
        batch_log_probs = []
        batch_returns = []
        batch_vols = []

        # Recolectar varios episodios antes de actualizar
        for _ in range(batch_size):
            probs, rewards, vol = generate_rl_data(model, instance_file, instance_number)

            # Calcular retornos descontados
            discounted_r = compute_discounted_returns(rewards, gamma)

            # Guardar log_probs y retornos
            log_probs = torch.log(torch.clamp(torch.stack(probs), 1e-6, 1.0))
            batch_log_probs.append(log_probs)
            batch_returns.append(discounted_r)
            batch_vols.append(vol)

        # Actualizar modelo con promedio de episodios
        loss = update_params(optimizer, batch_log_probs, batch_returns)

        # Métricas
        mean_vol = sum(batch_vols) / len(batch_vols)
        vol_arr.append(mean_vol)
        loss_arr.append(float(loss))

        print(f"[Ep {i+batch_size}] Volume Ratio: {mean_vol:.4f}\tLoss: {loss:.4f}")

    return vol_arr, loss_arr