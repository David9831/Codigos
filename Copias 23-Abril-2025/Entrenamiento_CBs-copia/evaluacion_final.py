# evaluacion_CB_tension.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from IEEE_33_Bus_System_CB import IEEE33BusSystem
from Actor_Critico_Buffer_CB import Actor # Asegúrate que Actor está definido aquí
import os
import glob
from datetime import datetime
import pandapower as pp

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- Hiperparámetros (deben coincidir con los del entrenamiento) ---
# !!! Revisa estos valores para que coincidan EXACTAMENTE con los de Entrenamiento_Actor_CB.py !!!
STATE_DIM = 4  # Dimensión del estado usada en el entrenamiento (matches env.STATE_DIM)
ACTION_DIM = 4 # Número de acciones discretas (0, 1, 2, 3) usado en el entrenamiento (matches env.ACTION_DIM)
HIDDEN_DIM = 256 # Dimensión oculta usada en el entrenamiento

# --- Configuración ---
MODEL_BASE_DIR = "modelo" # Carpeta base donde se guardan los modelos por fecha
NUM_EVAL_EPISODES = 1     # Cuántos días completos simular (normalmente 1 es suficiente para ver el patrón)
TARGET_BUS_INDEX = 6      # Índice del bus cuya tensión queremos monitorizar (Bus 6)

# --- Dispositivo ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

def find_latest_model_path(base_dir):
    """Encuentra la ruta al archivo actor_model.pth más reciente en subdirectorios con fecha."""
    try:
        # Listar todos los subdirectorios en la carpeta base
        date_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

        # Filtrar directorios que parecen fechas YYYY-MM-DD (opcional pero más robusto)
        valid_date_dirs = []
        for d in date_dirs:
            try:
                datetime.strptime(d, "%Y-%m-%d")
                valid_date_dirs.append(d)
            except ValueError:
                continue # Ignorar directorios que no coinciden con el formato

        if not valid_date_dirs:
            print(f"Error: No se encontraron directorios con formato YYYY-MM-DD en '{base_dir}'.")
            return None

        # Ordenar los directorios por fecha (el más reciente al final)
        latest_dir = sorted(valid_date_dirs)[-1]
        latest_model_dir = os.path.join(base_dir, latest_dir)
        model_path = os.path.join(latest_model_dir, "actor_model.pth")

        if os.path.exists(model_path):
            print(f"Encontrado modelo más reciente en: {model_path}")
            return model_path
        else:
            print(f"Error: Se encontró el directorio '{latest_model_dir}', pero no el archivo 'actor_model.pth' dentro.")
            return None

    except FileNotFoundError:
        print(f"Error: El directorio base '{base_dir}' no existe.")
        return None
    except Exception as e:
        print(f"Error inesperado al buscar el modelo: {e}")
        return None

def evaluate_actor_voltage(actor, env, num_episodes=1, target_bus=6):
    """
    Evalúa el actor, registra y grafica la tensión en un bus específico y las acciones.
    """
    actor.eval()  # Poner el actor en modo de evaluación
    all_episode_voltages = []
    all_episode_actions = []
    horas = list(range(1, 25)) # Horas del día

    print(f"\n--- Iniciando Evaluación ({num_episodes} episodio(s)) ---")

    with torch.no_grad():  # Desactivar el cálculo de gradientes durante la evaluación
        for episode in range(num_episodes):
            print(f"--- Episodio de Evaluación {episode + 1}/{num_episodes} ---")
            # Nota: El reset del capacitor se maneja dentro del bucle de horas
            # según el patrón de Entrenamiento_Actor_CB.py
            # env.reset() # Reset general si fuera necesario al inicio del episodio

            episode_voltages = []
            episode_actions = []

            # Simular un día completo (24 horas/steps)
            for step, hora in enumerate(horas):
                # Resetear estado base y aplicar carga horaria (como en entrenamiento)
                try:
                    # Resetear capacitor a 0 antes de cada hora (si es el comportamiento deseado)
                    env.net.shunt.at[env.controlled_shunt_idx, 'step'] = 0
                    pp.runpp(env.net) # Ejecutar flujo para que el estado inicial refleje el reset
                except Exception as e:
                    print(f"Advertencia: No se pudo resetear el capacitor en la hora {hora}. Error: {e}")

                env.reset() # Resetea las cargas a su valor base (según tu código de entorno)
                env.update_loads(hora) # Aplica el factor de carga para la hora actual

                # Obtener estado actual
                state = env.get_state()
                if state is None or np.all(state == 0): # Manejar caso de no convergencia en get_state
                    print(f"Advertencia: Estado inválido obtenido en hora {hora}. Saltando paso.")
                    # Podrías añadir un valor placeholder o manejarlo de otra forma
                    episode_voltages.append(np.nan) # Usar NaN si el estado es inválido
                    episode_actions.append(np.nan)
                    continue # Pasar a la siguiente hora

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

                # Obtener acción del actor
                action_tensor, _ = actor.get_action(state_tensor)
                action_value = action_tensor.item() # Obtener el valor entero de la acción

                # Aplicar acción en el entorno
                # step() aplica la acción, ejecuta runpp, calcula recompensa (no usada aquí), y obtiene next_state (no usado aquí)
                _, _, _ = env.step(action_value)

                # Registrar la tensión resultante en el bus objetivo DESPUÉS de la acción
                try:
                    # Asegurarse que los resultados del flujo están disponibles
                    if not env.net.res_bus.empty:
                        voltage_target_bus = env.net.res_bus.vm_pu.at[target_bus]
                        episode_voltages.append(voltage_target_bus)
                        episode_actions.append(action_value)
                        # print(f"  Hora {hora}: Estado={state}, Acción={action_value}, Voltaje Bus {target_bus}={voltage_target_bus:.4f}")
                    else:
                         print(f"Advertencia: res_bus vacío después de env.step() en hora {hora}. Voltaje no registrado.")
                         episode_voltages.append(np.nan)
                         episode_actions.append(action_value) # Guardar la acción aunque el voltaje falle

                except (KeyError, IndexError):
                    print(f"Advertencia: No se pudo obtener la tensión del bus {target_bus} en la hora {hora}.")
                    episode_voltages.append(np.nan) # Usar NaN si hay error al leer voltaje
                    episode_actions.append(action_value)
                except Exception as e:
                    print(f"Error inesperado al leer voltaje en hora {hora}: {e}")
                    episode_voltages.append(np.nan)
                    episode_actions.append(action_value)


            all_episode_voltages.append(episode_voltages)
            all_episode_actions.append(episode_actions)
            print(f"--- Fin Episodio {episode + 1} ---")

    print("\n--- Evaluación Completa ---")

    # --- Graficar Resultados ---
    if not all_episode_voltages or not all_episode_actions:
        print("No se generaron datos para graficar.")
        return

    print("Generando gráfico...")
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Eje primario (Voltaje)
    color_v = 'tab:red'
    ax1.set_xlabel("Hora del día")
    ax1.set_ylabel(f"Tensión (pu) en Bus {target_bus}", color=color_v)
    for i, episode_data in enumerate(all_episode_voltages):
        ax1.plot(horas, episode_data, label=f"Voltaje Ep. {i+1}", color=color_v, marker='o', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color_v)
    ax1.grid(True, axis='y', linestyle=':')
    ax1.axhline(y=1.05, color=color_v, linestyle='--', linewidth=1, label='Límite Superior (1.05 pu)')
    ax1.axhline(y=0.95, color=color_v, linestyle='--', linewidth=1, label='Límite Inferior (0.95 pu)')
    ax1.set_ylim(0.90, 1.10) # Ajusta si es necesario
    ax1.set_xticks(horas) # Asegura que todas las horas se muestren

    # Eje secundario (Acción)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color_a = 'tab:blue'
    ax2.set_ylabel("Acción del Capacitor (Step)", color=color_a)
    for i, episode_data in enumerate(all_episode_actions):
        # Usar 'steps-post' para que la línea represente el valor HASTA el siguiente punto
        ax2.step(horas, episode_data, label=f"Acción Ep. {i+1}", where='post', color=color_a, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_a)
    # Ajustar los ticks y límites del eje Y para las acciones discretas
    # Asumiendo que ACTION_DIM=3 significa acciones 0, 1, 2
    possible_actions = list(range(ACTION_DIM))
    ax2.set_yticks(possible_actions)
    ax2.set_ylim(-0.5, ACTION_DIM - 0.5) # Margen visual

    # Título y Leyenda
    plt.title(f"Evaluación del Actor: Tensión en Bus {target_bus} y Acciones del Capacitor")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes) # Combina leyendas
    fig.tight_layout()  # Ajusta el layout para prevenir solapamiento
    plt.grid(True)
    plt.show()

# --- Bloque Principal ---
if __name__ == "__main__":
    # 1. Crear el entorno
    env = IEEE33BusSystem()
    print("Entorno IEEE33BusSystem creado.")

    # 2. Encontrar y cargar el modelo del Actor más reciente
    latest_model_path = find_latest_model_path(MODEL_BASE_DIR)

    if latest_model_path:
        # Crear una instancia del modelo Actor
        loaded_actor = Actor(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)

        try:
            # Cargar los parámetros (state_dict)
            loaded_actor.load_state_dict(torch.load(latest_model_path, map_location=device))
            print(f"Modelo del Actor cargado exitosamente desde: {latest_model_path}")

            # 3. Evaluar el actor cargado
            evaluate_actor_voltage(loaded_actor, env, num_episodes=NUM_EVAL_EPISODES, target_bus=TARGET_BUS_INDEX)

        except FileNotFoundError:
            print(f"Error Crítico: El archivo del modelo '{latest_model_path}' no fue encontrado (esto no debería pasar si find_latest_model_path funcionó).")
        except Exception as e:
            print(f"Error al cargar el modelo o durante la evaluación: {e}")
            # Podrías querer imprimir el traceback completo para depurar:
            # import traceback
            # traceback.print_exc()
    else:
        print("No se pudo encontrar un modelo para evaluar. Asegúrate de que la carpeta 'modelo' exista y contenga subcarpetas YYYY-MM-DD con 'actor_model.pth'.")

    print("\nScript de evaluación finalizado.")
