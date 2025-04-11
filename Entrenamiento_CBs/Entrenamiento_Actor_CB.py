import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandapower as pp
from IEEE_33_Bus_System_CB import IEEE33BusSystem
from Actor_Critico_Buffer_CB import Actor, Critic, ReplayBuffer
import os
from tqdm import tqdm as bar


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
STATE_DIM = 16            # Dimensión del estado (ajustar según tu entorno)
ACTION_DIM = 6           # Número de taps del transformador (0-32)
HIDDEN_DIM = 256         # Dimensión de las capas ocultas
BUFFER_CAPACITY = 10000  # Capacidad del buffer Ajustado a un mes
BATCH_SIZE = 256         #tamaño del batch ajustado a un dia
NUM_EPISODES = 500
ALPHA = 0.02
GAMMA = 0.99             # Factor de descuento
TAU = 0.005              # Para la actualización suave de los críticos objetivo
LR_ACTOR = 1e-4          # Tasa de aprendizaje del Actor
LR_CRITIC = 1e-3         # Tasa de aprendizaje de los Críticos
# Inicializar el entorno
env = IEEE33BusSystem()
#----------------Inicializar el Actor y los Críticos
actor = Actor(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
critic1 = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
critic2 = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
# Inicializar los Críticos objetivo (copias de los Críticos)
target_critic1 = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
target_critic2 = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
target_critic1.load_state_dict(critic1.state_dict())  # Copiar parámetros
target_critic2.load_state_dict(critic2.state_dict())  # Copiar parámetros
#-----------------Inicializar el buffer de replay
buffer = ReplayBuffer(BUFFER_CAPACITY)
#-----------------Optimizadores
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=LR_CRITIC)
critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=LR_CRITIC)
#-----------------Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor.to(device)
critic1.to(device)
critic2.to(device)
target_critic1.to(device)
target_critic2.to(device)
#-----------------Función para actualizar los Críticos objetivo
def update_target_networks():
    for param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    for param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
#-----------------Entrenamiento del Actor
def train_actor():
    episode_rewards = []
    critic1_losses = []   # Lista para guardar las pérdidas de critic1
    critic2_losses = []   # Lista para guardar las pérdidas de critic2
    actor_losses = []     # Lista para guardar las pérdidas del actor
    all_actions = []      # Lista para guardar todas las acciones tomadas
    for episode in range(NUM_EPISODES):
        try:
            env.net.shunt.at[0, 'step'] = 0
            pp.runpp(env.net)
            #print("Banco de capacitores reseteado y flujo de potencia ejecutado.")
        except IndexError:
            print("Advertencia: No se encontró ningún shunt en la red para resetear.")
        except Exception as e:
            print(f"Error al resetear capacitor o ejecutar flujo de potencia: {e}")
        print(f"Iniciando episodio {episode + 1}...")
        env.reset()
        episode_reward = 0.0
        episode_actions = []
        for hora in range (1,25):
            env.reset()
            env.update_loads(hora)
            state = env.get_state()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_tensor, log_prob = actor.get_action(state_tensor)
            action_value = action_tensor.item()
            next_state, reward, done = env.step(action_value)
            buffer.push(state, action_value, reward, next_state, done)
            state = next_state
            episode_reward += reward  
            episode_actions.append(action_value) 
            if len(buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
                states = torch.FloatTensor(states).to(device)
                actions = torch.FloatTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)
                # --- Actualizar los Críticos ---
                with torch.no_grad():
                    # Muestrear una acción del Actor para el siguiente estado
                    next_actions, next_log_probs = actor.get_action(next_states)
                    target_q1 = target_critic1(next_states, next_actions)
                    target_q2 = target_critic2(next_states, next_actions)
                    target_q = torch.min(target_q1, target_q2) - ALPHA * next_log_probs
                    target_q = rewards + GAMMA * (1 - dones) * target_q
    
                # Calcular los valores Q actuales
                current_q1 = critic1(states, actions.unsqueeze(-1))
                current_q2 = critic2(states, actions.unsqueeze(-1))
    
                # Calcular la pérdida de los Críticos
                critic1_loss = F.mse_loss(current_q1, target_q)
                critic2_loss = F.mse_loss(current_q2, target_q)
    
                # Guardar las pérdidas
                critic1_losses.append(critic1_loss.item())
                critic2_losses.append(critic2_loss.item())
    
                # Actualizar los Críticos
                critic1_optimizer.zero_grad()
                critic1_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic1.parameters(), max_norm=1)
                critic1_optimizer.step()
                critic2_optimizer.zero_grad()
                critic2_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic2.parameters(), max_norm=1)
                critic2_optimizer.step()
                #------------------Actualizar el Actor ---
                new_actions, log_probs = actor.get_action(states)
                q1_new = critic1(states, new_actions)
                q2_new = critic2(states, new_actions)
                actor_loss = (ALPHA * log_probs - torch.min(q1_new, q2_new)).mean()
                actor_losses.append(actor_loss.item()) # Guardar la perdida del actor
                #------------------Actualizar el Actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
    
                #------------------Actualizar los Críticos Objetivo
                update_target_networks()
        # Guardar la recompensa y el episodio
        episode_rewards.append(episode_reward)
        all_actions.append(episode_actions)
        print(f"Episodio {episode + 1}/{NUM_EPISODES}, Recompensa: {episode_reward}")

    # Guardar el modelo entrenado del Actor y los criticos
    #torch.save(actor.state_dict(), "actor_model.pth")
    #torch.save(critic1.state_dict(), "critic1_model.pth")
    #torch.save(critic2.state_dict(), "critic2_model.pth")
    #print("Modelo del Actor y los criticos guardado.")

    return episode_rewards, critic1_losses, critic2_losses, actor_losses, all_actions

# Ejecutar el entrenamiento
if __name__ == "__main__":
    rewards, critic1_losses, critic2_losses, actor_losses, all_actions = train_actor()

    # Graficar las pérdidas de los críticos
    plt.figure(figsize=(10, 5))
    plt.plot(critic1_losses, label="Critic 1 Loss")
    plt.plot(critic2_losses, label="Critic 2 Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Critic Losses during Training")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Graficar las pérdidas del actor
    plt.figure(figsize=(10, 5))
    plt.plot(actor_losses, label="Actor Loss", color="green")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Actor Loss during Training")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Graficar el comportamiento de la recompensa después del entrenamiento ---
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Recompensa por Episodio", color="blue")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title("Comportamiento de la Recompensa durante el Entrenamiento")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Opcional: Graficar la recompensa promedio móvil ---
    def moving_average(data, window_size):
        """Calcula la media móvil de una lista de datos."""
        if len(data) < window_size:
            return data
        
        cumsum = np.cumsum(np.insert(data, 0, 0)) 
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    window_size = 10  # Tamaño de la ventana para la media móvil
    moving_avg_rewards = moving_average(rewards, window_size)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Recompensa por Episodio", color="blue", alpha=0.5)  # Recompensa original en un color más claro
    plt.plot(range(window_size - 1, len(rewards)), moving_avg_rewards, label=f"Media Móvil (Ventana = {window_size})", color="red")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title("Comportamiento de la Recompensa y Media Móvil durante el Entrenamiento")
    plt.legend()
    plt.grid(True)
    plt.show()
