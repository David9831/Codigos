import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandapower as pp
from IEEE_33_Bus_System_CB import IEEE33BusSystem
from Actor_Critico_Buffer_CB import Actor, Critic, ReplayBuffer
import os
from datetime import date
from tqdm import tqdm as bar


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Inicializar el entorno PRIMERO para obtener sus dimensiones
env = IEEE33BusSystem()

STATE_DIM = env.STATE_DIM        # Dimensión del estado (obtenida del entorno)
ACTION_DIM = env.ACTION_DIM      # Número de acciones discretas (obtenida del entorno)
HIDDEN_DIM = 256          # Dimensión de las capas ocultas
BUFFER_CAPACITY = 50000   # Capacidad del buffer Ajustado a un mes
BATCH_SIZE = 64          #tamaño del batch ajustado a un dia
NUM_EPISODES = 500        # Número de episodios de entrenamiento
#ALPHA = 0.02
ALPHA = 0.005086739257704386
GAMMA = 0.9678940829218391              # Factor de descuento
TAU = 0.04637435935730879             # Para la actualización suave de los críticos objetivo
LR_ACTOR = 0.0005869010009196598           # Tasa de aprendizaje del Actor
LR_CRITIC = 0.00038831497146954045         # Tasa de aprendizaje de los Críticos
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
#-----------------Función para actualizar los Críticos objetivo---------------------------
def update_target_networks():
    for param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    for param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
#-----------------Entrenamiento del Actor-------------------------------------------------
def train_actor():
    episode_rewards = []
    critic1_losses = []   # Lista para guardar las pérdidas de critic1
    critic2_losses = []   # Lista para guardar las pérdidas de critic2
    actor_losses = []     # Lista para guardar las pérdidas del actor
    all_actions = []      # Lista para guardar todas las acciones tomadas
    for episode in range(NUM_EPISODES):
        # Resetear el entorno al inicio de cada episodio
        env.reset() # <-- Mover aquí
        try:
            env.net.shunt.at[env.CONTROLLED_SHUNT_INDEX, 'step'] = 0 # Usar el índice correcto
            pp.runpp(env.net)
            #print("Banco de capacitores reseteado y flujo de potencia ejecutado.")
        except IndexError:
            print("Advertencia: No se encontró ningún shunt en la red para resetear.")
        print(f"Iniciando episodio {episode + 1}...")
        env.reset()
        episode_reward = 0.0
        episode_actions = []
        for hora in range (1,25):
            #env.reset()
            state = env.get_state()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_tensor, log_prob = actor.get_action(state_tensor)
            action_value = action_tensor.item()
            next_state, reward, done = env.step(action_value)
            buffer.push(state, action_value, reward, next_state, done)
            #state = next_state
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
    
                #------------------Actualizar los Críticos Objetivo------------------------------
                update_target_networks()
        # Guardar la recompensa y el episodio
        episode_rewards.append(episode_reward)
        all_actions.append(episode_actions)
        print(f"Episodio {episode + 1}/{NUM_EPISODES}, Recompensa: {episode_reward}")

    # Guardar el modelo entrenado del Actor y los criticos
    # Obtener fecha actual y formatearla
    today_str = date.today().strftime("%Y-%m-%d") # Formato YYYY-MM-DD

    # Definir directorios
    base_save_dir = "modelo"
    save_dir = os.path.join(base_save_dir, today_str)

    # Crear el directorio si no existe (incluyendo el directorio base 'modelo')
    # exist_ok=True evita un error si el directorio ya existe
    os.makedirs(save_dir, exist_ok=True)
    print(f"Creando/Verificando directorio de guardado: {save_dir}")

    # Definir rutas completas para los archivos del modelo
    actor_save_path = os.path.join(save_dir, "actor_model.pth")
    critic1_save_path = os.path.join(save_dir, "critic1_model.pth")
    critic2_save_path = os.path.join(save_dir, "critic2_model.pth")

    # Guardar los modelos en el directorio específico
    torch.save(actor.state_dict(), actor_save_path)
    torch.save(critic1.state_dict(), critic1_save_path)
    torch.save(critic2.state_dict(), critic2_save_path)
    print(f"Modelos del Actor y los críticos guardados en: {save_dir}")

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
