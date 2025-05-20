import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandapower as pp
from IEEE_33_Bus_System_CB import IEEE33BusSystem
env = IEEE33BusSystem
#Entradas del actor: "state_dim" <----- dimension del estado
#"hidden_dim" <---- Cantidad de neuronas de la capa profunda 

class Actor(nn.Module):   #nn.Module (clase base para todos los módulos de redes neuronales)
                          #state_dim se define en la linea 26 de IEEE_34_Bus_System
                          #action_dim se define en la linea 57 de IEEE_34_Bus_System

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        #Capas lineales
        self.fc1 = nn.Linear(state_dim, hidden_dim)    #nn.Linear se aplica una transformacion lineal a los datos entrantes
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)   
        self.fc3 = nn.Linear(hidden_dim, action_dim)   # salida de los logits para acciones discretas

    def forward(self, state):                          #propaga un estado a través de las 2 redes para obtener los logits de las acciones
                                                       # Como entrada de la capa 1 tenemos el estado del sistema "state"
        x = F.relu(self.fc1(state))                    # "State" <---- se define en la linea 153 del codigo IEEE_34_Bus_System
        x = F.relu(self.fc2(x))                        # Se usa la funcion de activacion relu para las dos primeras capas 
        logits = self.fc3(x)                           # En la ultima capa se proporcionan lo Logits 
        return logits   
    
    def get_action(self, state):
        logits=self.forward(state)                          #Logits de las acciones
        probs= F.softmax(logits, dim=-1)                    # convertir logits a probalidades
        action_dist=torch.distributions.Categorical(probs)  # Distribución categorica sobre las acciones (cada acción posible tiene una cierta probabilidad de ser elegida)
        action=action_dist.sample()                         # se muestra una accion aleatoria de esta distribución
        log_prob=action_dist.log_prob(action)               # (Log_prob) qué tan probable fue elegir esa acción
                                                            # Asegurar que la acción tenga la dimensión correcta
        action = action.unsqueeze(-1)                       # (batch_size, 1)
        return action, log_prob
        

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)  # representar acciones discretas en un espacio continuo de mayor dimensión
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)                           # devuelve un unico valor Q(s,a)
    
    def forward(self, state, action):
        action_embed = self.action_embedding(action.long().squeeze(-1))  # convierte la acción discreta en un vector denso
        # Procesar el estado
        state_out = F.relu(self.fc1(state))                              #Se extraen las caracteristicas en las capas 1 y 2 de la red
        state_out = F.relu(self.fc2(state_out))
        # Combinar estado y acción
        q_value = self.fc3(state_out + action_embed)                     # Sumar estado y acción para calcular el valor Q correspondiente
        return q_value.squeeze(-1)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Seleccionar indices aleatorios del buffer
        indices=np.random.choice(len(self.buffer), batch_size, replace=False)
        # Extraer las experiencias correspondientes a los indices seleccionados
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))        

        return states, actions, rewards, next_states, dones
    
    def print_buffer(self):
        print("Contenido del buffer:")
        for i, experience in enumerate(self.buffer):
            if experience is not None:
                state, action, reward, next_state, done = experience
                print(f" Experience{i}:")
                print(f"  State: {state}")
                print(f"  Action: {action}")
                print(f"  Reward: {reward}")
                print(f"  Next State: {next_state}")
                print(f"  Done: {done}")
            else:
                print(f"Experience {i}: None")
    
    def __len__(self):
        return len(self.buffer)
    
