import os # <--- Añadir esta línea
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # <--- Añadir esta línea

import torch
import numpy as np
import torch.nn.functional as F 
from IEEE_34_Bus_System_OLTC import IEEE33BusSystem
from Actor_Critico_Buffer_OLTC import Actor, Critic, ReplayBuffer
import matplotlib.pyplot as plt # <--- Añadir importación para graficar
indices_buses_recompensa=None
BUFFER_CAPACITY = 10000
BATCH_SIZE = 512
NUM_EPISODES = 500
GAMMA = 0.99            # Factor de descuento
TAU = 0.005             # Para la actualización suave de los críticos objetivo
ALPHA = 0.0005          # Coeficiente de entropía
LR_ACTOR = 3e-5         # Tasa de aprendizaje del Actor
LR_CRITIC = 1e-4        # Tasa de aprendizaje de los Críticos

# Inicializar el entorno
env = IEEE33BusSystem(reward_bus_indices=indices_buses_recompensa)

# Hiperparámetros dinámicos del entorno
STATE_DIM = env.state_dim         # Dimensión del estado
ACTION_DIM = env.action_dim       # Número de acciones (taps)
HIDDEN_DIM = 256                  # Dimensión de las capas ocultas

# Inicializar el Actor y los Críticos
actor = Actor(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
critic1 = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
critic2 = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)

# Inicializar los Críticos objetivo (copias de los Críticos)
target_critic1 = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
target_critic2 = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
target_critic1.load_state_dict(critic1.state_dict())  # Copiar parámetros
target_critic2.load_state_dict(critic2.state_dict())  # Copiar parámetros

# Inicializar el buffer de replay
buffer = ReplayBuffer(BUFFER_CAPACITY)

# Optimizadores
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=LR_CRITIC)
critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=LR_CRITIC)

#Dispositivo
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor.to(device)
critic1.to(device)
critic2.to(device)
target_critic1.to(device)
target_critic2.to(device)

# Función para actualizar los Críticos objetivo
def update_target_networks():
    for param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    for param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# Entrenamiento del Actor
def train_actor():
    episode_rewards = []
    # Listas para almacenar las pérdidas en cada paso de actualización
    all_critic1_losses = []
    all_critic2_losses = []
    all_actor_losses = []

    for episode in range(NUM_EPISODES):
        print(f"Iniciando episodio {episode+1}...")
        state = env.reset()  # Reiniciar el entorno
        episode_reward = 0.0
        step_count = 0

        for step in range(24):
            step_count +=1
            hora=step_count                                       #-----Inicia en la hora 1 
            env.update_loads(hora)                                #-----Actualiza las cargas según la demanda horaria 
            state=env.get_state()                                 #-----Se obitene el estado del sistema según la demanda horaria 
            state_tensor = torch.FloatTensor(state).unsqueeze(0)           #-----El estado se vuelve un tensor 
            action, log_prob = actor.get_action(state_tensor.to(device))     #-----Se obtiene la acción del Actor
            next_state, reward, done = env.step(action)           #-----Se ejecuta la acción del Actor en el entorno
            #print(f"Paso {step_count}: Acción = {action}, Recompensa = {reward}, Done = {done},estado={next_state}")

            # Almacenar la experiencia en el buffer
            buffer.push(state, action.cpu().numpy(), reward, next_state, done)

            # Actualizar el estado y la recompensa acumulada
            state = next_state
            episode_reward += reward 

            # Entrenar el agente si hay suficientes experiencias en el buffer
            if len(buffer) >= BATCH_SIZE:
                # Muestrear un batch de experiencias del buffer
                states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

                # Convertir a tensores de PyTorch
                states = torch.FloatTensor(states).to(device)
                actions = torch.FloatTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                # --- Actualizar los Críticos ---
                with torch.no_grad():
                    # Muestrear una acción del Actor para el siguiente estado
                    next_actions, next_log_probs = actor.get_action(next_states)

                    # Calcular los valores Q objetivo
                    target_q1 = target_critic1(next_states, next_actions)
                    target_q2 = target_critic2(next_states, next_actions)
                    target_q = torch.min(target_q1, target_q2) - ALPHA * next_log_probs   # verificar en los articulos
                    target_q = rewards + GAMMA * (1 - dones) * target_q

                # Calcular los valores Q actuales
                current_q1 = critic1(states, actions)
                current_q2 = critic2(states, actions)

                # Calcular la pérdida de los Críticos
                critic1_loss = F.mse_loss(current_q1, target_q)
                critic2_loss = F.mse_loss(current_q2, target_q)
                # Guardar las pérdidas
                all_critic1_losses.append(critic1_loss.item())
                all_critic2_losses.append(critic2_loss.item())

                # Actualizar los Críticos
                critic1_optimizer.zero_grad()
                critic1_loss.backward()
                critic1_optimizer.step()

                critic2_optimizer.zero_grad()
                critic2_loss.backward()
                critic2_optimizer.step()

                # --- Actualizar el Actor ---
                new_actions, log_probs = actor.get_action(states)
                q1_new = critic1(states, new_actions)
                q2_new = critic2(states, new_actions)
                actor_loss = (ALPHA * log_probs - torch.min(q1_new, q2_new)).mean()
                # Guardar la pérdida del actor
                all_actor_losses.append(actor_loss.item())

                # Actualizar el Actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # --- Actualizar los Críticos Objetivo ---
                update_target_networks()

             # Verificar si el episodio ha terminado
            #if done:
            #    break
            #    print(f"Episodio {episode+1} terminado despues de {step_count}")

        #Guardar la recompensa y el episodio
        episode_rewards.append(episode_reward)
        #print(env.get_state())
        Vnodos,_,_=env.variables_interes()
        print(f"Episodio {episode + 1}/{NUM_EPISODES}, Recompensa: {episode_reward}")
        #print(Vnodos)


    # Guardar el modelo entrenado del Actor
    #torch.save(actor.state_dict(), "actor_model.pth")
    #print("Modelo del Actor guardado.")

    return episode_rewards, all_critic1_losses, all_critic2_losses, all_actor_losses

# Ejecutar el entrenamiento
if __name__ == "__main__":
    episode_rewards, critic1_losses, critic2_losses, actor_losses = train_actor()

    # Graficar resultados
    plt.figure(figsize=(15, 10))

    # Gráfica de Recompensas por Episodio
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Recompensa por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Total del Episodio')
    plt.grid(True)

    # Gráfica de Pérdida del Crítico 1
    plt.subplot(2, 2, 2)
    plt.plot(critic1_losses)
    plt.title('Pérdida del Crítico 1')
    plt.xlabel('Paso de Actualización de Red')
    plt.ylabel('Pérdida (MSE)')
    plt.grid(True)

    # Gráfica de Pérdida del Crítico 2
    plt.subplot(2, 2, 3)
    plt.plot(critic2_losses)
    plt.title('Pérdida del Crítico 2')
    plt.xlabel('Paso de Actualización de Red')
    plt.ylabel('Pérdida (MSE)')
    plt.grid(True)

    # Gráfica de Pérdida del Actor
    plt.subplot(2, 2, 4)
    plt.plot(actor_losses)
    plt.title('Pérdida del Actor')
    plt.xlabel('Paso de Actualización de Red')
    plt.ylabel('Pérdida')
    plt.grid(True)

    plt.tight_layout() # Ajusta automáticamente los parámetros del subplot para dar un buen ajuste.
    plt.show()