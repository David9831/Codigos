import pandapower as pp
# import pandapower.plotting as pp_plot # No usado en este snippet
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from IEEE_33_Bus_System_CB import IEEE33BusSystem
from Actor_Critico_Buffer_CB import Actor

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# --- Hiperparámetros ---
STATE_DIM = 3
ACTION_DIM = 4
HIDDEN_DIM = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define el dispositivo
# --- Crear Instancias ---
env = IEEE33BusSystem()
actor_instance = Actor(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
# --- ******** NUEVO: RESETEAR EL BANCO DE CAPACITORES A TAP 0 ******** ---
print("Reseteando banco de capacitores a tap 0...")
# Accede al DataFrame de shunts y establece la columna 'step' a 0
# Asumiendo que el capacitor es el primer (o único) shunt en la red (índice 0)
# Si tienes más shunts y quieres resetearlos todos, usa env.net.shunt['step'] = 0
try:
    env.net.shunt.at[0, 'step'] = 0 # Establece el paso del shunt en índice 0 a 0
    # Es crucial ejecutar un flujo de potencia para que este cambio se refleje
    # en los resultados que obtendrá get_state()
    pp.runpp(env.net)
    print("Banco de capacitores reseteado y flujo de potencia ejecutado.")
except IndexError:
    print("Advertencia: No se encontró ningún shunt en la red para resetear.")
except Exception as e:
    print(f"Error al resetear capacitor o ejecutar flujo de potencia: {e}")
# --- Preparar Estado ---
hora = 3
# Ahora obtenemos el estado DESPUÉS de haber reseteado el capacitor y actualizado las cargas
state_tuple = env.get_state()
state_tensor = torch.FloatTensor(state_tuple).unsqueeze(0).to(device)
# --- Obtener Acción y Probabilidades ---
with torch.no_grad(): # Importante para inferencia
    # Llama a get_action en la INSTANCIA del actor
    action_tensor, log_prob, action_probabilities = actor_instance.get_action(state_tensor)
# --- Procesar y Mostrar Resultados ---
action_value = action_tensor.item() # Obtener el valor entero de la acción seleccionada
probs_np = action_probabilities.squeeze(0).cpu().numpy()
print("-" * 30)
print(f"Hora: {hora}")
print(f"Estado Inicial (Tupla) (con capacitor en 0): {state_tuple}")
# print(f"Estado (Tensor): {state_tensor}") # Descomentar si quieres ver el tensor
print("-" * 30)
# --- MOSTRAR LA ACCIÓN SELECCIONADA ---
print(f"Acción Seleccionada por el Actor: {action_value}")
print(f"Log Probabilidad de la Acción Seleccionada: {log_prob.item():.4f}")
print("-" * 30)
print("Probabilidades para cada acción posible:")
for i, prob in enumerate(probs_np):
    print(f"  Acción {i}: {prob:.4f}") # Imprime la probabilidad de cada acción formateada
print("-" * 30)
print(f"Suma de probabilidades: {np.sum(probs_np):.4f}") # Verificar que suman ~1
print("-" * 30)

