import optuna
import torch
import numpy as np
from IEEE_33_Bus_System_CB import IEEE33BusSystem
from Actor_Critico_Buffer_CB import Actor, Critic, ReplayBuffer

def train_sac(params):
    # === Inicialización del entorno ===
    env = IEEE33BusSystem()
    state_dim = 4
    action_dim = 4
    hidden_dim = params["hidden_dim"]

    # === Inicializar redes ===
    actor = Actor(state_dim, action_dim, hidden_dim)
    critic1 = Critic(state_dim, action_dim, hidden_dim)
    critic2 = Critic(state_dim, action_dim, hidden_dim)
    target_critic1 = Critic(state_dim, action_dim, hidden_dim)
    target_critic2 = Critic(state_dim, action_dim, hidden_dim)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    # === Replay buffer ===
    buffer = ReplayBuffer(10000)

    # === Optimizadores ===
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=params["actor_lr"])
    critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=params["critic_lr"])
    critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=params["critic_lr"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor.to(device)
    critic1.to(device)
    critic2.to(device)
    target_critic1.to(device)
    target_critic2.to(device)

    # === Función auxiliar para actualizar redes objetivo ===
    def update_targets():
        for p, tp in zip(critic1.parameters(), target_critic1.parameters()):
            tp.data.copy_(params["tau"] * p.data + (1 - params["tau"]) * tp.data)
        for p, tp in zip(critic2.parameters(), target_critic2.parameters()):
            tp.data.copy_(params["tau"] * p.data + (1 - params["tau"]) * tp.data)

    total_rewards = []

    for episode in range(30):  # Solo 10 episodios por prueba
        env.reset()
        episode_reward = 0

        for hora in range(1, 25):
            env.update_loads(hora)
            state = env.get_state()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            action_tensor, log_prob = actor.get_action(state_tensor)
            action_value = action_tensor.item()

            next_state, reward, done = env.step(action_value)
            buffer.push(state, action_value, reward, next_state, done)
            episode_reward += reward

            if len(buffer) >= params["batch_size"]:
                states, actions, rewards, next_states, dones = buffer.sample(params["batch_size"])
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                with torch.no_grad():
                    next_actions, next_log_probs = actor.get_action(next_states)
                    q1_target = target_critic1(next_states, next_actions)
                    q2_target = target_critic2(next_states, next_actions)
                    q_target = torch.min(q1_target, q2_target) - params["alpha"] * next_log_probs
                    q_target = rewards + params["gamma"] * (1 - dones) * q_target

                q1 = critic1(states, actions.unsqueeze(-1))
                q2 = critic2(states, actions.unsqueeze(-1))
                critic1_loss = torch.nn.functional.mse_loss(q1, q_target)
                critic2_loss = torch.nn.functional.mse_loss(q2, q_target)

                critic1_optimizer.zero_grad()
                critic1_loss.backward()
                critic1_optimizer.step()
                critic2_optimizer.zero_grad()
                critic2_loss.backward()
                critic2_optimizer.step()

                new_actions, log_probs = actor.get_action(states)
                q1_new = critic1(states, new_actions)
                q2_new = critic2(states, new_actions)
                actor_loss = (params["alpha"] * log_probs - torch.min(q1_new, q2_new)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                update_targets()

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)

# === Función de Optuna ===
def objective(trial):
    params = {
        "actor_lr": trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True),
        "critic_lr": trial.suggest_float("critic_lr", 1e-5, 1e-3, log=True),
        "alpha": trial.suggest_float("alpha", 1e-4, 0.1, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "tau": trial.suggest_float("tau", 0.005, 0.05),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256])
    }
    return train_sac(params)

# === Ejecutar búsqueda ===
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # Puedes aumentar a 50 o más

    print("Mejores hiperparámetros:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
