import torch
import numpy as np
import matplotlib.pyplot as plt
from IEEE_33_Bus_System_CB import IEEE33BusSystem
from Actor_Critico_Buffer_CB import Actor, Critic
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Hiperparámetros (deben coincidir con los del entrenamiento)
STATE_DIM = 3
ACTION_DIM = 4
HIDDEN_DIM = 256

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_actor(actor, env, num_episodes=1):
    """
    Evalúa el actor en el entorno y grafica la tensión en el nodo 6 y las acciones tomadas.

    Args:
        actor: El modelo del actor entrenado.
        env: El entorno IEEE33BusSystem.
        num_episodes: Número de episodios para evaluar.
    """
    actor.eval()  # Poner el actor en modo de evaluación
    all_episode_voltages = []
    all_episode_actions = []

    with torch.no_grad():  # Desactivar el cálculo de gradientes
        for episode in range(num_episodes):
            print(f"Evaluando episodio {episode + 1}...")
            state = env.reset()
            episode_voltages = []
            episode_actions = []
            for step in range(24):
                hora = step + 1
                env.reset()
                env.update_loads(hora)
                state = env.get_state()
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, _ = actor.get_action(state_tensor)
                action_value = action.item() # Get the action value
                env.step(action_value)

                # Obtener la tensión del nodo 6
                voltage_bus_6 = env.net.res_bus.vm_pu.at[6]
                episode_voltages.append(voltage_bus_6)
                episode_actions.append(action_value)  # Guardar la acción tomada

            all_episode_voltages.append(episode_voltages)
            all_episode_actions.append(episode_actions)

    # Graficar las tensiones del nodo 6 y las acciones después de que terminen todos los episodios
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Graficar las tensiones
    for episode_idx, episode_voltages in enumerate(all_episode_voltages):
        ax1.plot(episode_voltages, label=f"Tensión Episodio {episode_idx + 1}", color=f"C{episode_idx}")

    ax1.set_xlabel("Step (Dentro del Episodio)")
    ax1.set_ylabel("Tensión (pu) en el Nodo 6", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.set_title(f"Tensión en el Nodo 6 (Todos los Episodios)")
    ax1.grid(True)

    # Crear un segundo eje y para las acciones
    ax2 = ax1.twinx()
    for episode_idx, episode_actions in enumerate(all_episode_actions):
        ax2.plot(episode_actions, label=f"Acción Episodio {episode_idx + 1}", color=f"C{episode_idx}", linestyle="--")

    ax2.set_ylabel("Acción", color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")
    ax2.set_yticks(np.arange(0, ACTION_DIM))
    ax2.set_ylim([0,ACTION_DIM])

    # Combinar las leyendas
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")

    fig.tight_layout()
    plt.show()
    return all_episode_voltages, all_episode_actions

if __name__ == "__main__":
    # Crear el entorno
    env = IEEE33BusSystem()

    # Crear una instancia del modelo Actor (debe ser la misma arquitectura que el modelo guardado)
    loaded_actor = Actor(STATE_DIM, ACTION_DIM, HIDDEN_DIM)

    # Cargar los parámetros del modelo desde el archivo
    try:
        loaded_actor.load_state_dict(torch.load("actor_model.pth"))
        print("Modelo cargado exitosamente.")
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'actor_model.pth'. Asegúrate de haber entrenado y guardado el modelo.")
        exit()

    # Poner el modelo en modo de evaluación
    loaded_actor.eval()

    # Mover el modelo al dispositivo correcto (CPU o GPU)
    loaded_actor.to(device)

    # Evaluar el actor
    all_episode_voltages, all_episode_actions = evaluate_actor(loaded_actor, env, num_episodes=1)
