import torch
import torch.nn as nn
import torch.optim as optim
import random
from copy import deepcopy
import numpy as np
from ..env import Environment
from ..models.base.encoder_decoder_pe import EncoderDecoderPEModel
from ..training import get_predictions

env = Environment()

# ============================================================
# --- Representación inmutable del estado para DQN ---
# ============================================================
class DQNState:
    def __init__(self, state, valid_actions):
        """
        Captura el estado actual de manera estática:
        - X_src: bloques del contenedor
        - X_tgt: acciones válidas (en forma de vectores)
        - placed: indicadores de bloques colocados
        - coords: coordenadas actuales de los bloques
        """
        self.X_src = torch.tensor([state.get_blocks()], dtype=torch.float32)
        self.placed = torch.tensor([state.get_placed()], dtype=torch.float32)
        self.coords = torch.tensor([state.get_coords()], dtype=torch.float32)

        # Acciones válidas -> batch de candidatos
        action_vecs = [a.action_vec for a in valid_actions]
        self.X_tgt = torch.tensor([action_vecs], dtype=torch.float32)

    def clone(self):
        """Devuelve una copia profunda del estado"""
        new = DQNState.__new__(DQNState)
        new.X_src = self.X_src.clone()
        new.X_tgt = self.X_tgt.clone()
        new.placed = self.placed.clone()
        new.coords = self.coords.clone()
        return new

# ============================================================
# --- DQN Agent adaptado al entorno de colocación de bloques ---
# ============================================================
class DQNAgent:
    def __init__(self, model, instance_file, instance_number,
                 epsilon=0.2, epsilon_decay=0.95, gamma=0.99,
                 learning_rate=1e-4, batch_size=16, target_model_update_freq=5,
                 memory_size=1000):

        self.instance_file = instance_file
        self.instance_number = instance_number
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_model_update_freq = target_model_update_freq
        self.memory_size = memory_size
        self.memory = []

        self.model = model
        self.target_model = deepcopy(model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        torch.manual_seed(42)
        random.seed(42)

    # -----------------------------------------------------------
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # -----------------------------------------------------------
    def select_action(self, dqn_state, greedy=False):
        """
        Política epsilon-greedy o determinista.
        """
        with torch.no_grad():
            q_values = get_predictions(
                self.model,
                dqn_state.X_src,
                dqn_state.X_tgt,
                dqn_state.placed,
                dqn_state.coords,
                apply_softmax=False
            ).squeeze(0)  # [num_valid_actions]

        if greedy or random.random() > self.epsilon:
            action_idx = torch.argmax(q_values).item()
        else:
            action_idx = random.randint(0, q_values.size(0) - 1)

        return action_idx, q_values[action_idx].item()

    # -----------------------------------------------------------
    def remember(self, dqn_state, action_idx, reward, next_dqn_state, done):
        self.memory.append((dqn_state, action_idx, reward, next_dqn_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    # -----------------------------------------------------------
    def sample_batch(self):
        return random.sample(self.memory, self.batch_size)

    # -----------------------------------------------------------
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.sample_batch()
        losses = []

        for dqn_state, action_idx, reward, next_dqn_state, done in batch:
            # Q(s,a)
            q_values = get_predictions(
                self.model,
                dqn_state.X_src,
                dqn_state.X_tgt,
                dqn_state.placed,
                dqn_state.coords,
                apply_softmax=False
            ).squeeze(0)
            q_value = q_values[action_idx]

            # Q_target(s', a')
            with torch.no_grad():
                next_q_values = get_predictions(
                    self.target_model,
                    next_dqn_state.X_src,
                    next_dqn_state.X_tgt,
                    next_dqn_state.placed,
                    next_dqn_state.coords,
                    apply_softmax=False
                ).squeeze(0)
                target_q = reward + self.gamma * torch.max(next_q_values) * (1 - int(done))

            # MSE
            loss = self.criterion(q_value, target_q)
            losses.append(loss)

        # Backprop acumulado
        loss_batch = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss_batch.backward()
        self.optimizer.step()

        return loss_batch.item()

    # -----------------------------------------------------------
    def learn(self, episodes=20):
        for ep in range(episodes):
            state = env.initial_state(self.instance_file, self.instance_number, w=8)
            done = False
            total_reward = 0.0
            last_loss = None  # 👈 agregado

            while True:
                valid_actions = env.get_valid_actions(
                    state,
                    add_block_index=isinstance(self.model, EncoderDecoderPEModel)
                )
                if len(valid_actions) == 0:
                    break

                dqn_state = DQNState(state, valid_actions)

                # Elegir acción
                action_idx, q_val = self.select_action(dqn_state)
                selected_action = valid_actions[action_idx]

                # Ejecutar acción
                reward = env.state_transition(state, selected_action)
                total_reward += reward

                # Crear siguiente estado inmutable
                next_valid = env.get_valid_actions(
                    state,
                    add_block_index=isinstance(self.model, EncoderDecoderPEModel)
                )
                next_dqn_state = DQNState(state, next_valid) if len(next_valid) > 0 else None
                done = len(next_valid) == 0

                # Guardar transición (si hay next_state)
                if next_dqn_state:
                    self.remember(dqn_state.clone(), action_idx, reward, next_dqn_state.clone(), done)

                # Entrenamiento online
                loss = self.train_step()
                if loss is not None:
                    last_loss = loss 

                if done:
                    break

            # Actualización del modelo objetivo
            if (ep + 1) % self.target_model_update_freq == 0:
                self.update_target_model()
                self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)

            vol_ratio = state.get_volume_ratio()
            print(f"[Ep {ep+1}] Volume Ratio: {vol_ratio:.4f}\tEpsilon: {self.epsilon:.3f}\tLoss: {last_loss if last_loss is not None else 'N/A'}")

            state.close()