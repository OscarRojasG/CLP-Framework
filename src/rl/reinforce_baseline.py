from ..env import Environment
import torch
from torch import optim
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

env = Environment()

# ==========================================================
# 🔹 Generar datos de un episodio (actor-critic)
# ==========================================================
def generate_rl_data(model, instance_file, instance_number):
    log_probs = []
    rewards = []
    values = []

    state = env.initial_state(instance_file, instance_number, w=8)

    while True:
        try:
            add_block_index = True  # es EncoderDecoderPEModel
            valid_actions = env.get_valid_actions(state, add_block_index)
        except Exception as e:
            print("Error obteniendo acciones válidas:", e)
            state.close()
            raise

        if len(valid_actions) == 0:
            break

        # Convertir acciones a tensores
        action_vec = np.array([action.action_vec for action in valid_actions])
        X_tgt = torch.tensor([action_vec], dtype=torch.float32)

        X_src = torch.tensor([state.get_blocks()], dtype=torch.float32)
        placed = torch.tensor([state.get_placed()], dtype=torch.float32)
        coords = torch.tensor([state.get_coords()], dtype=torch.float32)

        # --- Forward del modelo actor-crítico ---
        logits, value = model(X_src, X_tgt, placed, coords)  # logits (1, N), value (1,)
        probs = F.softmax(logits.squeeze(0), dim=-1)
        dist = Categorical(probs)

        # Elegir acción
        action_idx = dist.sample()
        selected_action = valid_actions[action_idx]

        # Aplicar acción y obtener recompensa
        reward = env.state_transition(state, selected_action)

        # Guardar log_prob y valor
        log_probs.append(dist.log_prob(action_idx))
        values.append(value.squeeze(0))
        rewards.append(torch.tensor(reward, dtype=torch.float32))

    vol_ratio = state.get_volume_ratio()
    state.close()

    return log_probs, values, rewards, vol_ratio


# ==========================================================
# 🔹 Calcular retornos descontados
# ==========================================================
def compute_discounted_returns(rewards, gamma):
    discounted_returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_returns.insert(0, R)
    return torch.stack(discounted_returns)


# ==========================================================
# 🔹 Actualizar parámetros (Actor + Crítico)
# ==========================================================
def update_params(optimizer, log_probs, values, returns):
    values = torch.stack(values)
    log_probs = torch.stack(log_probs)

    # Baseline = valores del crítico
    advantages = returns - values.detach()

    # Actor loss (policy gradient con ventaja)
    actor_loss = -(log_probs * advantages).mean()

    # Critic loss (MSE entre retorno y valor predicho)
    critic_loss = F.mse_loss(values, returns)

    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
    optimizer.step()

    return loss, actor_loss, critic_loss


# ==========================================================
# 🔹 Entrenamiento REINFORCE con baseline (Actor-Critic)
# ==========================================================
def reinforce_with_baseline(model, instance_file, instance_number, eps=200, lr=1e-4, gamma=0.99, batch_size=4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    vol_arr, loss_arr = [], []

    model.train()

    for i in range(0, eps, batch_size):
        batch_loss, batch_actor, batch_critic, batch_vols = [], [], [], []

        for _ in range(batch_size):
            log_probs, values, rewards, vol = generate_rl_data(model, instance_file, instance_number)
            returns = compute_discounted_returns(rewards, gamma)

            loss, actor_loss, critic_loss = update_params(optimizer, log_probs, values, returns)
            batch_loss.append(loss.item())
            batch_actor.append(actor_loss.item())
            batch_critic.append(critic_loss.item())
            batch_vols.append(vol)

        mean_vol = np.mean(batch_vols)
        mean_loss = np.mean(batch_loss)
        mean_actor = np.mean(batch_actor)
        mean_critic = np.mean(batch_critic)

        vol_arr.append(mean_vol)
        loss_arr.append(mean_loss)

        print(
            f"[Ep {i+batch_size}] "
            f"Volume Ratio: {mean_vol:.4f}\t"
            f"Total Loss: {mean_loss:.4f}\t"
            f"Actor Loss: {mean_actor:.4f}\t"
            f"Critic Loss: {mean_critic:.4f}"
        )

    return vol_arr, loss_arr