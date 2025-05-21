# c:/Users/David/Documents/Codigos/Entrenamiento_CBs/evaluacion_comparativa_CB.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
from IEEE_33_Bus_System_CB import IEEE33BusSystem
from Actor_Critico_Buffer_CB import Actor
import os
from datetime import datetime
import glob

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- Hiperparámetros (deben coincidir con los del entrenamiento) ---
env_eval = IEEE33BusSystem() # Crear una instancia para obtener dimensiones
STATE_DIM = env_eval.STATE_DIM  # Dimensión del estado (V, P, Q en bus objetivo, Cap Step, Tap Pos)
ACTION_DIM = env_eval.ACTION_DIM # Número de acciones discretas combinadas
ACTION_DIM_CAP = env_eval.ACTION_DIM_CAP # Número de acciones del capacitor
ACTION_DIM_TAP = env_eval.ACTION_DIM_TAP # Número de acciones del tap
HIDDEN_DIM = 256 # Dimensión oculta de las redes neuronales

del env_eval # Eliminar la instancia temporal
# --- Configuración de Evaluación ---
MODEL_BASE_DIR = "modelo"
TARGET_BUS_IDX = 6      # Índice del bus para monitorizar tensión (e.g., Bus 6)
CONTROLLED_SHUNT_IDX = 0 # Índice del shunt controlado
FIXED_ACTION_BASELINE = 0 # Acción fija para el escenario base (e.g., capacitor en step 0)
HORAS_DIA = list(range(1, 25)) # Horas del 1 al 24

# --- Dispositivo ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

def find_latest_model_path(base_dir):
    """Encuentra la ruta al archivo actor_model.pth más reciente."""
    try:
        date_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        valid_date_dirs = []
        for d in date_dirs:
            try:
                datetime.strptime(d, "%Y-%m-%d")
                valid_date_dirs.append(d)
            except ValueError:
                continue
        if not valid_date_dirs:
            return None
        latest_dir = sorted(valid_date_dirs)[-1]
        model_path = os.path.join(base_dir, latest_dir, "actor_model.pth")
        return model_path if os.path.exists(model_path) else None
    except Exception:
        return None

def run_simulation_day(env, actor_model=None, fixed_baseline_config=None,
                       target_bus_idx=TARGET_BUS_IDX, controlled_shunt_idx=CONTROLLED_SHUNT_IDX,
                       controlled_trafo_idx=0): # Add trafo index
    """
    Simula un día completo (24 horas) y registra métricas.
    Si actor_model no es None, usa el actor.
    Sino, si fixed_baseline_config es (fixed_cap_step, fixed_tap_pos), usa esas acciones fijas.
    Sino, usa cap step 0 y tap neutral por defecto.
    """
    env.reset() # Resetea cargas a base, step_count a 0.

    # Asegurar estado inicial de los dispositivos (ej. Cap a 0, Tap a neutral) al inicio del día
    initial_capacitor_step = 0

    initial_tap_pos = 16 # Asumiendo tap neutral 16 como en la definición del trafo

    if fixed_baseline_config is not None and actor_model is None: # Si es baseline con config fija
        initial_capacitor_step, initial_tap_pos = fixed_baseline_config

    # Set initial capacitor step
    try:
        env.net.shunt.at[controlled_shunt_idx, 'step'] = initial_capacitor_step
    except Exception as e:
        print(f"Advertencia: No se pudo resetear el capacitor a {initial_capacitor_step} al inicio del día. {e}")
    # Set initial tap position
    try:
        env.net.trafo.at[controlled_trafo_idx, 'tap_pos'] = initial_tap_pos
    except Exception as e:
        print(f"Advertencia: No se pudo resetear el tap a {initial_tap_pos} al inicio del día. {e}")

    try:
        env.net.shunt.at[controlled_shunt_idx, 'step'] = initial_capacitor_step
        pp.runpp(env.net, algorithm='nr', calculate_voltage_angles=True)
    except Exception as e:
        print(f"Advertencia: No se pudo resetear el capacitor a {initial_capacitor_step} al inicio del día. {e}")

    voltages_hourly = []
    actions_hourly = []
    cap_steps_hourly = [] # Track capacitor steps
    losses_hourly = []
    tap_pos_hourly = [] # Track tap positions
    
    if actor_model:
        print("Iniciando simulación de un día (Actor)...")
    elif fixed_baseline_config:
        print(f"Iniciando simulación de un día (Baseline Fijo Cap={fixed_baseline_config[0]}, Tap={fixed_baseline_config[1]})...")
    else:
        print("Iniciando simulación de un día (Modo desconocido/defecto)...")

    for hora_idx, hora_actual in enumerate(HORAS_DIA):
        env.update_loads(hora_actual) # Actualiza cargas para la hora actual

        # Para el primer paso (hora 1), el estado se basa en el capacitor reseteado y cargas de hora 1.
        # Para pasos subsiguientes, el estado se basa en el resultado del paso anterior.
        # Es crucial que pp.runpp() se haya ejecutado para tener res_bus actualizado.
        # env.step() se encarga de esto. Si es el primer paso, pp.runpp() ya corrió arriba.
        # Si no es el primer paso, env.step() del ciclo anterior actualizó res_bus.

        current_state = env.get_state()

        action_to_take_combined = 0 # Default action index
        current_cap_step = 0        # Default capacitor step for logging
        current_tap_pos = initial_tap_pos # Default tap position for logging

        if actor_model:
            actor_model.eval()
            with torch.no_grad():
                if np.all(current_state == 0) and hora_idx > 0 : # Si el estado es inválido (ej. no convergencia previa)
                    print(f"  Hora {hora_actual}: Estado inválido, usando acción combinada por defecto 0 (Cap 0, Tap Keep).")
                    action_to_take_combined = 0 # Tomar una acción segura o predeterminada
                elif current_state.shape[0] != STATE_DIM:
                    print(f"  Hora {hora_actual}: Advertencia: Dimensión del estado ({current_state.shape[0]}) no coincide con STATE_DIM ({STATE_DIM}). Usando acción por defecto 0.")
                    action_to_take_combined = 0
                else:
                    state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)
                    action_tensor, _ = actor_model.get_action(state_tensor)
                    action_to_take_combined = action_tensor.item()
            
            _next_state, _reward, done = env.step(action_to_take_combined)
            # Decode the combined action to get individual device actions for logging
            current_cap_step = action_to_take_combined // ACTION_DIM_TAP # Use global ACTION_DIM_TAP
            current_tap_pos = int(env.net.trafo.at[controlled_trafo_idx, 'tap_pos'])

        elif fixed_baseline_config is not None:
            # Baseline mode: directly set device states and run power flow
            cap_action_to_take, tap_pos_to_set = fixed_baseline_config
            env.net.shunt.at[controlled_shunt_idx, 'step'] = cap_action_to_take
            env.net.trafo.at[controlled_trafo_idx, 'tap_pos'] = tap_pos_to_set
            try:
                pp.runpp(env.net, algorithm='nr', calculate_voltage_angles=True)
                _next_state = env.get_state() # Get state after runpp
                _reward = 0 # No reward calculation needed for baseline run
                done = env.is_done()
            except Exception as e:
                print(f"  Hora {hora_actual}: Advertencia: Power flow failed in baseline run. {e}")
                _next_state = env.get_state() # Attempt to get state
                _reward = -10 # Penalize failure
                done = True # Consider episode done on failure
            
            action_to_take_combined = -1 # Placeholder for logging, not an agent action
            current_cap_step = cap_action_to_take
            current_tap_pos = tap_pos_to_set
        else: # Use the combined action from the actor
            # Fallback: neither actor nor baseline config provided. Use default action.
            print(f"  Hora {hora_actual}: Ni actor ni configuración baseline provistos. Usando acción combinada por defecto 0.")
            action_to_take_combined = 0 # Default: Cap 0, Tap Keep
            _next_state, _reward, done = env.step(action_to_take_combined)
            # Decode the combined action to get individual device actions for logging
            current_cap_step = action_to_take_combined // ACTION_DIM_TAP # Use global ACTION_DIM_TAP
            current_tap_pos = int(env.net.trafo.at[controlled_trafo_idx, 'tap_pos'])

        # Registrar resultados DESPUÉS de la acción y el flujo de potencia
        if not env.net.res_bus.empty:
            voltages_hourly.append(env.net.res_bus.vm_pu.at[target_bus_idx])
            current_total_losses = env.net.res_line.pl_mw.sum() if not env.net.res_line.empty else np.nan
            losses_hourly.append(current_total_losses)
        else:
            print(f"  Hora {hora_actual}: res_bus vacío después de env.step(). Voltaje/Pérdidas no registrados.")
            voltages_hourly.append(np.nan)
            losses_hourly.append(np.nan)
        
        actions_hourly.append(action_to_take_combined) # Log the combined action index
        cap_steps_hourly.append(current_cap_step) # Log the resulting cap step
        tap_pos_hourly.append(current_tap_pos) # Log the resulting tap position
        # print(f"  Hora {hora_actual}: Acción={action_to_take}, Voltaje={voltages_hourly[-1]:.4f} pu, Pérdidas={losses_hourly[-1]:.4f} MW")

        if done and hora_idx < len(HORAS_DIA) - 1:
            print(f"  Simulación terminada prematuramente en hora {hora_actual} por env.is_done().")
            # Rellenar el resto con NaN para mantener la longitud de las listas
            remaining_steps = len(HORAS_DIA) - 1 - hora_idx
            voltages_hourly.extend([np.nan] * remaining_steps)
            actions_hourly.extend([np.nan] * remaining_steps) # O la última acción tomada
            cap_steps_hourly.extend([np.nan] * remaining_steps)
            tap_pos_hourly.extend([np.nan] * remaining_steps)
            losses_hourly.extend([np.nan] * remaining_steps)
            break
            
    return voltages_hourly, actions_hourly, cap_steps_hourly, tap_pos_hourly, losses_hourly

def plot_comparison_results(horas, results_agent, results_baseline, target_bus_idx, fixed_baseline_config):
    volt_agent, actions_agent, cap_steps_agent, tap_pos_agent, losses_agent = results_agent
    # Unpack all 5 elements from results_baseline, even if some are not used directly here
    volt_base, _actions_base, _cap_steps_base, _tap_pos_base, losses_base = results_baseline

    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # 1. Voltaje
    axs[0].plot(horas, volt_agent, label=f"Agente - Voltaje Bus {target_bus_idx}", color='blue', marker='o', linestyle='-')
    axs[0].plot(horas, volt_base, label=f"Baseline (Cap@{fixed_baseline_config[0]}, Tap@{fixed_baseline_config[1]}) - Voltaje Bus {target_bus_idx}", color='gray', marker='x', linestyle='--')
    axs[0].axhline(1.05, color='red', linestyle=':', linewidth=1, label='Límite Superior (1.05 pu)')
    axs[0].axhline(0.95, color='red', linestyle=':', linewidth=1, label='Límite Inferior (0.95 pu)')
    axs[0].set_ylabel("Tensión (pu)")
    axs[0].set_ylim(0.90, 1.10)
    axs[0].grid(True, linestyle=':')
    axs[0].set_title(f"Comparación de Rendimiento: Agente vs Baseline (Cap@{fixed_baseline_config[0]}, Tap@{fixed_baseline_config[1]})")

    # 2. Pérdidas Activas Totales
    axs[1].plot(horas, losses_agent, label="Agente - Pérdidas Totales", color='green', marker='o', linestyle='-')
    axs[1].plot(horas, losses_base, label=f"Baseline (Cap@{fixed_baseline_config[0]}, Tap@{fixed_baseline_config[1]}) - Pérdidas Totales", color='lightgreen', marker='x', linestyle='--')
    axs[1].set_ylabel("Pérdidas Activas Totales (MW)")
    axs[1].grid(True, linestyle=':')

    # 3. Acciones de los dispositivos (Agente y Baseline)
    ax_actions = axs[2]
    # Plot Agent Capacitor Steps
    ax_actions.step(horas, cap_steps_agent, label="Agente - Step Capacitor", where='post', color='purple', zorder=10)
    ax_actions.set_ylabel("Acción del Capacitor (Step)")
    ax_actions.set_yticks(list(range(ACTION_DIM_CAP))) # Usar global ACTION_DIM_CAP
    ax_actions.set_ylim(-0.5, ACTION_DIM_CAP - 0.5) # Usar global ACTION_DIM_CAP
    ax_actions.legend(loc='upper left')
    ax_actions.grid(True, linestyle=':')

    # Plot Agent Tap Positions on a twin axis
    ax_actions_twin = ax_actions.twinx()
    ax_actions_twin.step(horas, tap_pos_agent, label="Agente - Posición Tap", where='post', color='darkorange', linestyle='-')
    ax_actions_twin.set_ylabel("Posición Tap Transformador", color='darkorange')
    ax_actions_twin.tick_params(axis='y', labelcolor='darkorange')
    # Set y-ticks based on possible tap positions (e.g., 0 to 33)
    # Assuming tap_min=0, tap_max=33 from env setup
    ax_actions_twin.set_yticks(np.arange(0, 34, 5)) # Example: ticks every 5 positions
    ax_actions_twin.set_ylim(-1, 34) # Example limits
    ax_actions_twin.legend(loc='upper right')

    # Add Baseline fixed actions to the plot
    # Baseline Capacitor Step (plotted on primary y-axis)
    ax_actions.plot(horas, [fixed_baseline_config[0]] * len(horas), label=f"Baseline - Cap Step Fijo ({fixed_baseline_config[0]})", color='gray', linestyle='dotted', zorder=5)
    # Baseline Tap Position (plotted on secondary y-axis)
    ax_actions_twin.plot(horas, [fixed_baseline_config[1]] * len(horas), label=f"Baseline - Tap Pos Fija ({fixed_baseline_config[1]})", color='silver', linestyle='dotted', zorder=5)

    # Combine legends
    lines, labels = ax_actions.get_legend_handles_labels()
    lines2, labels2 = ax_actions_twin.get_legend_handles_labels()
    # Place combined legend on the first subplot for overall clarity
    axs[0].legend(lines + lines2, labels + labels2, loc='lower left', ncol=2)
    axs[1].legend(loc='best') # Individual legend for losses
    # For the third subplot, legends are already set for ax_actions and ax_actions_twin
    
    plt.xlabel("Hora del día")
    plt.xticks(horas)
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar para título general si se añade
    plt.show()

def print_summary(results_agent, results_baseline, fixed_action_baseline):
    volt_agent, _actions_agent_combined, cap_steps_agent, tap_pos_agent, losses_agent = results_agent
    volt_base, _, _cap_steps_base, _tap_pos_base, losses_base = results_baseline


    # Filtrar NaNs para cálculos
    volt_agent_valid = [v for v in volt_agent if not np.isnan(v)]
    losses_agent_valid = [l for l in losses_agent if not np.isnan(l)]
    volt_base_valid = [v for v in volt_base if not np.isnan(v)]
    losses_base_valid = [l for l in losses_base if not np.isnan(l)]
    
    num_switching_ops = 0
    if len(cap_steps_agent) > 1:
        for i in range(len(cap_steps_agent) - 1):
            if not np.isnan(cap_steps_agent[i]) and not np.isnan(cap_steps_agent[i+1]) and \
               cap_steps_agent[i] != cap_steps_agent[i+1]:
                num_switching_ops +=1
            
    num_tap_ops = 0
    if len(tap_pos_agent) > 1:
        for i in range(len(tap_pos_agent) - 1):
            if not np.isnan(tap_pos_agent[i]) and not np.isnan(tap_pos_agent[i+1]) and \
               tap_pos_agent[i] != tap_pos_agent[i+1]:
                num_tap_ops +=1

    print("\n--- Resumen de la Evaluación ---")
    print(f"{'Métrica':<30} | {'Agente':<15} | {'Baseline (Cap@{fixed_action_baseline[0]}, Tap@{fixed_action_baseline[1]})':<25}") # Use fixed_action_baseline tuple
    print("-" * 70)
    if volt_agent_valid:
        print(f"{'Voltaje Promedio (pu)':<30} | {np.mean(volt_agent_valid):<15.4f} | {np.mean(volt_base_valid) if volt_base_valid else 'N/A':<20.4f}")
        print(f"{'Voltaje Mínimo (pu)':<30} | {np.min(volt_agent_valid):<15.4f} | {np.min(volt_base_valid) if volt_base_valid else 'N/A':<20.4f}")
        print(f"{'Voltaje Máximo (pu)':<30} | {np.max(volt_agent_valid):<15.4f} | {np.max(volt_base_valid) if volt_base_valid else 'N/A':<20.4f}")
    else:
        print(f"{'Voltajes (pu)':<30} | {'N/A':<15} | {'N/A':<20}")

    if losses_agent_valid and losses_base_valid : # Check both are valid for sum
        print(f"{'Pérdidas Totales Acum. (MWh)':<30} | {np.sum(losses_agent_valid):<15.4f} | {np.sum(losses_base_valid) if losses_base_valid else 'N/A':<20.4f}")
    else:
        print(f"{'Pérdidas Totales Acum. (MWh)':<30} | {'N/A' if not losses_agent_valid else np.sum(losses_agent_valid):<15.4f} | {'N/A' if not losses_base_valid else np.sum(losses_base_valid):<20.4f}")
    print(f"{'Operaciones de Switch (Cap)':<30} | {num_switching_ops:<15} | {'0 (fijo)':<20}") # Baseline Cap is fixed
    print(f"{'Operaciones de Tap (Trafo)':<30} | {num_tap_ops:<15} | {'0 (fijo)':<20}") # Baseline Tap is fixed     
    print("-" * 70)


if __name__ == "__main__":
    # 1. Crear el entorno
    env = IEEE33BusSystem()
    print("Entorno IEEE33BusSystem creado.")

    # 2. Encontrar y cargar el modelo del Actor
    actor_model_path = find_latest_model_path(MODEL_BASE_DIR)
    loaded_actor = None
    if actor_model_path:
        print(f"Encontrado modelo del actor en: {actor_model_path}")
        loaded_actor = Actor(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
        try:
            loaded_actor.load_state_dict(torch.load(actor_model_path, map_location=device))
            print("Modelo del Actor cargado exitosamente.")
        except Exception as e:
            print(f"Error al cargar el modelo del actor: {e}")
            loaded_actor = None # No usar actor si falla la carga
    else:
        print("Advertencia: No se encontró un modelo de actor entrenado. La simulación del agente no se ejecutará.")

    # 3. Ejecutar simulación con el Agente (si se cargó)
    results_agent = ([np.nan]*len(HORAS_DIA), [np.nan]*len(HORAS_DIA), [np.nan]*len(HORAS_DIA), [np.nan]*len(HORAS_DIA), [np.nan]*len(HORAS_DIA)) # Initialize with NaNs
    if loaded_actor:
        volt_agent, actions_agent, cap_steps_agent, tap_pos_agent, losses_agent = run_simulation_day(
            env,
            actor_model=loaded_actor,
            target_bus_idx=TARGET_BUS_IDX,
            controlled_shunt_idx=CONTROLLED_SHUNT_IDX,
            controlled_trafo_idx=0 # Assuming trafo index 0
        )
        results_agent = (volt_agent, actions_agent, cap_steps_agent, tap_pos_agent, losses_agent)


    # 4. Ejecutar simulación Baseline (Capacitor fijo)
    # Define baseline configuration: (fixed_cap_step, fixed_tap_pos)
    fixed_baseline_config = (FIXED_ACTION_BASELINE, 16) # Cap fixed at 0, Tap fixed at neutral (16)
    print(f"\nEjecutando simulación Baseline con capacitor fijo en step {fixed_baseline_config[0]} y tap fijo en posición {fixed_baseline_config[1]}...")
    volt_base, actions_base, cap_steps_base, tap_pos_base, losses_base = run_simulation_day(
        env,
        actor_model=None, # Sin actor
        fixed_baseline_config=fixed_baseline_config,
        target_bus_idx=TARGET_BUS_IDX,
        controlled_trafo_idx=0 # Assuming trafo index 0
    )
    results_baseline = (volt_base, actions_base, cap_steps_base, tap_pos_base, losses_base)

    # 5. Graficar y resumir resultados
    if loaded_actor or FIXED_ACTION_BASELINE is not None: # Solo graficar si hay algo que mostrar
        plot_comparison_results(HORAS_DIA, results_agent, results_baseline, TARGET_BUS_IDX, fixed_baseline_config) # Pass fixed_baseline_config
        print_summary(results_agent, results_baseline, fixed_baseline_config) # Pass fixed_baseline_config
    else:
        print("No hay datos para graficar o resumir.")

    print("\nScript de evaluación comparativa finalizado.")
