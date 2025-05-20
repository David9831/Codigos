import pandapower as pp
import numpy as np
import matplotlib.pyplot as plt
import pandapower.plotting as pp_plot
import torch.nn as nn
import torch.nn.functional as F 
import torch

class IEEE33BusSystem:
    def __init__(self, reward_bus_indices=None):
        self.net = pp.create_empty_network(f_hz=60,sn_mva=1)
        self.base_loads = [
            (0.1, 0.06),
            (0.09, 0.04), 
            (0.12, 0.08), 
            (0.06, 0.03), 
            (0.06, 0.02),
            (0.2, 0.1), 
            (0.2, 0.1), 
            (0.06, 0.02), 
            (0.06, 0.02), 
            (0.045, 0.03),
            (0.06, 0.035), 
            (0.06, 0.035), 
            (0.12, 0.08), 
            (0.06, 0.01), 
            (0.06, 0.02),
            (0.06, 0.02), 
            (0.09, 0.04), 
            (0.09, 0.04), 
            (0.09, 0.04), 
            (0.09, 0.04),
            (0.09, 0.04), 
            (0.09, 0.05), 
            (0.42, 0.2), 
            (0.42, 0.2), 
            (0.06, 0.025),
            (0.06, 0.025), 
            (0.06, 0.02), 
            (0.12, 0.07), 
            (0.2, 0.6), 
            (0.15, 0.07),
            (0.21, 0.1), 
            (0.06, 0.04)
        ]
        self.porcentaje_demanda = [
            0.28, 
            0.25,
            0.26, 
            0.25, 
            0.23, 
            0.44, 
            0.69, 
            0.44, 
            0.44, 
            0.36, 
            0.38, 
            0.51,
            0.4, 
            0.38, 
            0.37, 
            0.41, 
            0.49, 
            0.34, 
            0.61, 
            0.81, 
            1.0, 
            0.86, 
            0.69, 
            0.36
        ]
        self.perfil_voltaje = []
        self.step_count = 0
        self.MAX_STEPS = 24
        self.reward_bus_indices = reward_bus_indices # Guardar los índices de bus para la recompensa
        self.setup_network() # Configura la red
        
        # Inicializar state_dim correctamente
        # Primero obtenemos un estado inicial para determinar su longitud (dimensión)
        # Es importante correr un flujo de potencia aquí para que res_bus esté disponible.
        try:
            pp.runpp(self.net) 
        except Exception as e:
            print(f"Error inicializando la red durante el primer flujo de potencia: {e}")
            # Considerar manejar este error de forma más robusta si es necesario
        initial_state = self._get_raw_state() # Usar un método interno para evitar doble pp.runpp
        self.state_dim = len(initial_state) 

    def is_done(self):
        # Terminar el episodio si se alcanza el límite de pasos
        return self.step_count >= self.MAX_STEPS

    def setup_network(self):
        # Crear buses
        Vnhkv=110
        Vnlkv=12.666
        b0 = pp.create_bus(self.net, vn_kv=Vnhkv, name="Bus 0")
        b1 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 1")
        b2 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 2")
        b3 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 3")
        b4 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 4")
        b5 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 5")
        b6 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 6")
        b7 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 7")
        b8 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 8")
        b9 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 9")
        b10 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 10")
        b11 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 11")
        b12 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 12")
        b13 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 13")
        b14 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 14")
        b15 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 15")
        b16 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 16")
        b17 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 17")
        b18 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 18")
        b19 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 19")
        b20 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 20")
        b21 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 21")
        b22 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 22")
        b23 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 23")
        b24 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 24")
        b25 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 25")
        b26 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 26")
        b27 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 27")
        b28 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 28")
        b29 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 29")
        b30 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 30")
        b31 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 31")
        b32 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 32")
        b33 = pp.create_bus(self.net, vn_kv=Vnlkv, name="Bus 33")

        # Agregar generador en el nodo principal (slack)
        pp.create_ext_grid(self.net, bus=b0, vm_pu=1.0, name="Slack_Bus")
        # Agregar transformador
        pp.create_transformer_from_parameters(
            self.net, 
            hv_bus=b0, 
            lv_bus=b1, 
            sn_mva=5.5, 
            vn_hv_kv=110, 
            vn_lv_kv=12.666,
            pfe_kw=110, 
            i0_percent=0, 
            vk_percent=8, 
            vkr_percent=0.5, 
            tap_side="hv", 
            tap_neutral=16,
            tap_min=0, 
            tap_max=33, 
            tap_step_percent=1.25,            
            tap_pos=16,
            tap_changer_type = "ratio"
        )
        self.action_dim=33      #Cantidad de taps del transformador 
                                # Esto implica acciones 0-32. Tap_max es 33. La pos 33 no será alcanzable por el agente.
        # Agregar líneas de distribución
        a=0
        b=1
        # La siguiente línea es incorrecta, ya que b0 y b1 están conectados por el transformador. Se elimina.
        #pp.create_line_from_parameters(self.net, from_bus=b0, to_bus=b1, length_km=1, r_ohm_per_km=0.0922,x_ohm_per_km=0.0470, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.4930,x_ohm_per_km=0.2511, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b2, to_bus=b3, length_km=1, r_ohm_per_km=0.3660,x_ohm_per_km=0.1864, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b3, to_bus=b4, length_km=1, r_ohm_per_km=0.3811,x_ohm_per_km=0.1941, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b4, to_bus=b5, length_km=1, r_ohm_per_km=0.8190,x_ohm_per_km=0.7070, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b5, to_bus=b6, length_km=1, r_ohm_per_km=0.1872,x_ohm_per_km=0.6188, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b6, to_bus=b7, length_km=1, r_ohm_per_km=0.7114,x_ohm_per_km=0.2351, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b7, to_bus=b8, length_km=1, r_ohm_per_km=1.0300,x_ohm_per_km=0.7400, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b8, to_bus=b9, length_km=1, r_ohm_per_km=1.0440,x_ohm_per_km=0.7400, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b9, to_bus=b10, length_km=1, r_ohm_per_km=0.1966,x_ohm_per_km=0.0650, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b10, to_bus=b11, length_km=1, r_ohm_per_km=0.3744,x_ohm_per_km=0.1238, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b11, to_bus=b12, length_km=1, r_ohm_per_km=1.4680,x_ohm_per_km=1.1550, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b12, to_bus=b13, length_km=1, r_ohm_per_km=0.5416,x_ohm_per_km=0.7129, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b13, to_bus=b14, length_km=1, r_ohm_per_km=0.5910,x_ohm_per_km=0.5260, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b14, to_bus=b15, length_km=1, r_ohm_per_km=0.7463,x_ohm_per_km=0.5450, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b15, to_bus=b16, length_km=1, r_ohm_per_km=1.2890,x_ohm_per_km=1.7210, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b16, to_bus=b17, length_km=1, r_ohm_per_km=0.7320,x_ohm_per_km=0.5740, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b17, to_bus=b18, length_km=1, r_ohm_per_km=0.1640,x_ohm_per_km=0.1565, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b2, to_bus=b19, length_km=1, r_ohm_per_km=1.5042,x_ohm_per_km=1.3554, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b19, to_bus=b20, length_km=1, r_ohm_per_km=0.4095,x_ohm_per_km=0.4784, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b20, to_bus=b21, length_km=1, r_ohm_per_km=0.7089,x_ohm_per_km=0.9373, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b21, to_bus=b22, length_km=1, r_ohm_per_km=0.4512,x_ohm_per_km=0.3083, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b3, to_bus=b23, length_km=1, r_ohm_per_km=0.8980,x_ohm_per_km=0.7091, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b23, to_bus=b24, length_km=1, r_ohm_per_km=0.8960,x_ohm_per_km=0.7011, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b24, to_bus=b25, length_km=1, r_ohm_per_km=0.2030,x_ohm_per_km=0.1034, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b6, to_bus=b26, length_km=1, r_ohm_per_km=0.2842,x_ohm_per_km=0.1447, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b26, to_bus=b27, length_km=1, r_ohm_per_km=1.0590,x_ohm_per_km=0.9337, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b27, to_bus=b28, length_km=1, r_ohm_per_km=0.8042,x_ohm_per_km=0.7006, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b28, to_bus=b29, length_km=1, r_ohm_per_km=0.5075,x_ohm_per_km=0.2585, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b29, to_bus=b30, length_km=1, r_ohm_per_km=0.9744,x_ohm_per_km=0.9630, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b30, to_bus=b31, length_km=1, r_ohm_per_km=0.3105,x_ohm_per_km=0.3619, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b31, to_bus=b32, length_km=1, r_ohm_per_km=0.3410,x_ohm_per_km=0.5302, c_nf_per_km=a, max_i_ka=b)
        pp.create_line_from_parameters(self.net, from_bus=b32, to_bus=b33, length_km=1, r_ohm_per_km=0.0922,x_ohm_per_km=0.0470, c_nf_per_km=a, max_i_ka=b)



        # Agregar cargas en los nodos
        pp.create_load(self.net, bus=b1, p_mw=self.base_loads[0][0], q_mvar=self.base_loads[0][1], name="Load 1")
        pp.create_load(self.net, bus=b2, p_mw=self.base_loads[1][0], q_mvar=self.base_loads[1][1], name="Load 2")
        pp.create_load(self.net, bus=b3, p_mw=self.base_loads[2][0], q_mvar=self.base_loads[2][1], name="Load 3")
        pp.create_load(self.net, bus=b4, p_mw=self.base_loads[3][0], q_mvar=self.base_loads[3][1], name="Load 4")
        pp.create_load(self.net, bus=b5, p_mw=self.base_loads[4][0], q_mvar=self.base_loads[4][1], name="Load 5")
        pp.create_load(self.net, bus=b6, p_mw=self.base_loads[5][0], q_mvar=self.base_loads[5][1], name="Load 6")
        pp.create_load(self.net, bus=b7, p_mw=self.base_loads[6][0], q_mvar=self.base_loads[6][1], name="Load 7")
        pp.create_load(self.net, bus=b8, p_mw=self.base_loads[7][0], q_mvar=self.base_loads[7][1], name="Load 8")
        pp.create_load(self.net, bus=b9, p_mw=self.base_loads[8][0], q_mvar=self.base_loads[8][1], name="Load 9")
        pp.create_load(self.net, bus=b10, p_mw=self.base_loads[9][0], q_mvar=self.base_loads[9][1], name="Load 10")
        pp.create_load(self.net, bus=b11, p_mw=self.base_loads[10][0], q_mvar=self.base_loads[10][1], name="Load 11")
        pp.create_load(self.net, bus=b12, p_mw=self.base_loads[11][0], q_mvar=self.base_loads[11][1], name="Load 12")
        pp.create_load(self.net, bus=b13, p_mw=self.base_loads[12][0], q_mvar=self.base_loads[12][1], name="Load 13")
        pp.create_load(self.net, bus=b14, p_mw=self.base_loads[13][0], q_mvar=self.base_loads[13][1], name="Load 14")
        pp.create_load(self.net, bus=b15, p_mw=self.base_loads[14][0], q_mvar=self.base_loads[14][1], name="Load 15")
        pp.create_load(self.net, bus=b16, p_mw=self.base_loads[15][0], q_mvar=self.base_loads[15][1], name="Load 16")
        pp.create_load(self.net, bus=b17, p_mw=self.base_loads[16][0], q_mvar=self.base_loads[16][1], name="Load 17")
        pp.create_load(self.net, bus=b18, p_mw=self.base_loads[17][0], q_mvar=self.base_loads[17][1], name="Load 18")
        pp.create_load(self.net, bus=b19, p_mw=self.base_loads[18][0], q_mvar=self.base_loads[18][1], name="Load 19")
        pp.create_load(self.net, bus=b20, p_mw=self.base_loads[19][0], q_mvar=self.base_loads[19][1], name="Load 20")
        pp.create_load(self.net, bus=b21, p_mw=self.base_loads[20][0], q_mvar=self.base_loads[20][1], name="Load 21")
        pp.create_load(self.net, bus=b22, p_mw=self.base_loads[21][0], q_mvar=self.base_loads[21][1], name="Load 22")
        pp.create_load(self.net, bus=b23, p_mw=self.base_loads[22][0], q_mvar=self.base_loads[22][1], name="Load 23")
        pp.create_load(self.net, bus=b24, p_mw=self.base_loads[23][0], q_mvar=self.base_loads[23][1], name="Load 24")
        pp.create_load(self.net, bus=b25, p_mw=self.base_loads[24][0], q_mvar=self.base_loads[24][1], name="Load 25")
        pp.create_load(self.net, bus=b26, p_mw=self.base_loads[25][0], q_mvar=self.base_loads[25][1], name="Load 26")
        pp.create_load(self.net, bus=b27, p_mw=self.base_loads[26][0], q_mvar=self.base_loads[26][1], name="Load 27")
        pp.create_load(self.net, bus=b28, p_mw=self.base_loads[27][0], q_mvar=self.base_loads[27][1], name="Load 28")
        pp.create_load(self.net, bus=b29, p_mw=self.base_loads[28][0], q_mvar=self.base_loads[28][1], name="Load 29")
        pp.create_load(self.net, bus=b30, p_mw=self.base_loads[29][0], q_mvar=self.base_loads[29][1], name="Load 30")
        pp.create_load(self.net, bus=b31, p_mw=self.base_loads[30][0], q_mvar=self.base_loads[30][1], name="Load 31")
        pp.create_load(self.net, bus=b32, p_mw=self.base_loads[31][0], q_mvar=self.base_loads[31][1], name="Load 32")

    #----------Actualización de las cargas dependiendo de la demanda horaria ----------------------------
    def update_loads(self, hora_ciclo): # hora_ciclo se espera que sea de 1 a 24
        if not (1 <= hora_ciclo <= len(self.porcentaje_demanda)):
            print(f"Advertencia: hora_ciclo {hora_ciclo} fuera de rango para porcentaje_demanda (1-{len(self.porcentaje_demanda)}). Usando hora 1.")
            idx_demanda = 0 # Corresponde a self.porcentaje_demanda[0]
        else:
            idx_demanda = hora_ciclo - 1 # Convertir 1-24 a 0-23 para indexar
        
        current_demand_factor = self.porcentaje_demanda[idx_demanda]
        for i, (p_base, q_base) in enumerate(self.base_loads):
               self.net.load.at[i, 'p_mw'] = p_base * current_demand_factor
               self.net.load.at[i, 'q_mvar'] = q_base * current_demand_factor
        # La modificación de self.net.load es in-place, no es necesario devolverlo.
        
    #--------- Resetear el sistema de distribucion con las cargas iniciales------------------------------
    def reset(self):
        # Resetear cargas a sus valores base
        for i, (p_base, q_base) in enumerate(self.base_loads):
            self.net.load.at[i, 'p_mw'] = p_base
            self.net.load.at[i, 'q_mvar'] = q_base
        
        self.step_count = 0 # Resetear el contador de pasos del episodio

        # Opcional: Resetear la posición del tap a neutral si se desea un inicio consistente del tap
        self.net.trafo.loc[0, 'tap_pos'] = self.net.trafo.tap_neutral.iloc[0]

        # Correr flujo de potencia para tener un estado inicial consistente después del reset
        try:
            pp.runpp(self.net)
        except pp.LoadflowNotConverged:
            print("Advertencia: Loadflow no convergió durante el reset. El estado puede ser inconsistente.")
            # Aquí podrías devolver un estado de error o un estado por defecto si es necesario
        
        return self.get_state() # Devolver el estado inicial del entorno
    
    #--------- Función de recompensa---------------------------------------------------------------------
    def calculate_reward(self):
        cv = 1

        # Asegurarse de que los resultados del flujo de potencia estén disponibles
        if self.net.res_bus is None or 'vm_pu' not in self.net.res_bus or self.net.res_bus.empty:
            print("Advertencia: Resultados del flujo de potencia no disponibles en calculate_reward. Devolviendo penalización alta.")
            return -1000.0 # Devolver una penalización grande

        all_voltages_pu = self.net.res_bus.vm_pu

        if self.reward_bus_indices is not None and isinstance(self.reward_bus_indices, list) and len(self.reward_bus_indices) > 0:
            # Usar los índices de bus especificados por el usuario
            # Filtrar para asegurar que los índices sean válidos
            valid_indices = [idx for idx in self.reward_bus_indices if 0 <= idx < len(all_voltages_pu)]
            if len(valid_indices) != len(self.reward_bus_indices):
                print(f"Advertencia: Algunos índices de bus para recompensa eran inválidos. Se usarán solo los válidos: {valid_indices}")
            
            if not valid_indices:
                print("Advertencia: No hay índices de bus válidos especificados para recompensa. Usando todos los buses LV por defecto.")
                # Fallback al comportamiento por defecto si no quedan índices válidos
                voltajes_considerados = all_voltages_pu.iloc[1:self.net.bus.shape[0]]
            else:
                voltajes_considerados = all_voltages_pu.iloc[valid_indices]
        else:
            # Comportamiento por defecto: considerar todos los buses de baja tensión (índices 1 a 33)
            # Bus 0 es HV. Buses 1 hasta self.net.bus.shape[0]-1 son LV.
            voltajes_considerados = all_voltages_pu.iloc[1:self.net.bus.shape[0]]
        
        penalizacion_voltaje = 0
        # Penalización por voltajes bajos
        voltajes_bajos = voltajes_considerados[voltajes_considerados < 0.95]
        if not voltajes_bajos.empty:
            penalizacion_voltaje += np.sum(0.95 - voltajes_bajos)

        # Penalización por voltajes altos
        voltajes_altos = voltajes_considerados[voltajes_considerados > 1.05]
        if not voltajes_altos.empty:
            penalizacion_voltaje += np.sum(voltajes_altos - 1.05)
        
        penalizacion_voltaje *= cv
        recompensa = -penalizacion_voltaje
        return recompensa

    #----------- Definición de la acción del actor-----------------------------------------------------
    def step(self, action):
        # Manejar tipo de acción (tensor de PyTorch o entero/flotante)
        if isinstance(action, torch.Tensor):
            actual_tap_pos = action.item()
        elif isinstance(action, (int, float)):
            actual_tap_pos = int(action)
        else:
            raise TypeError(f"Tipo de acción {type(action)} no soportado. Debe ser Tensor, int o float.")

        # Validar que la acción esté dentro de los límites del tap
        tap_min_val = self.net.trafo.tap_min.iloc[0]
        tap_max_val = self.net.trafo.tap_max.iloc[0]
        if not (tap_min_val <= actual_tap_pos <= tap_max_val):
            print(f"Advertencia: Acción de tap {actual_tap_pos} fuera de rango ({tap_min_val}-{tap_max_val}). Se ajustará.")
            actual_tap_pos = np.clip(actual_tap_pos, tap_min_val, tap_max_val)

        self.step_count += 1 
        self.net.trafo.loc[0, 'tap_pos'] = actual_tap_pos

        # --- Información sobre el cambio de Tap y Ratio ---
        trafo_params = self.net.trafo.iloc[0]
        vn_hv_kv = trafo_params.vn_hv_kv
        vn_lv_kv = trafo_params.vn_lv_kv
        tap_neutral = trafo_params.tap_neutral
        tap_step_percent = trafo_params.tap_step_percent
        tap_side = trafo_params.tap_side

        nominal_ratio = vn_hv_kv / vn_lv_kv
        
        if tap_side == "hv":
            actual_ratio_calculated = nominal_ratio * (1 + (actual_tap_pos - tap_neutral) * (tap_step_percent / 100.0))
        elif tap_side == "lv":
            actual_ratio_calculated = nominal_ratio / (1 + (actual_tap_pos - tap_neutral) * (tap_step_percent / 100.0))
        else: # "neutral" o no especificado, sin cambio por tap
            actual_ratio_calculated = nominal_ratio

        #print(f"\n--- Info del Paso {self.step_count} (Acción de Tap: {action}) ---")
        #print(f"  Posición de Tap Aplicada: {actual_tap_pos}")
        #print(f"  Posición Neutral: {tap_neutral}, Paso: {tap_step_percent}%, Lado Tap: {tap_side}")
        #print(f"  Ratio Nominal (V_hv_nom / V_lv_nom): {vn_hv_kv:.3f}kV / {vn_lv_kv:.3f}kV = {nominal_ratio:.4f}")
        #print(f"  Relación de Transformación Teórica Calculada (V_hv_eff / V_lv_eff): {actual_ratio_calculated:.4f}")
        # --- Fin Información ---

        pp.runpp(self.net)

        # Voltajes resultantes en los terminales del transformador
        vm_hv_pu_trafo = self.net.res_trafo.vm_hv_pu.iloc[0]
        vm_lv_pu_trafo = self.net.res_trafo.vm_lv_pu.iloc[0]
        if vm_lv_pu_trafo != 0 and vn_lv_kv !=0 : # Evitar división por cero
            # Ratio_from_voltages = (Vhv_actual_terminal / Vlv_actual_terminal)
            # Vhv_actual_terminal = vm_hv_pu_trafo * vn_hv_kv (asumiendo vm_hv_pu es referido a vn_hv_kv del trafo)
            # Vlv_actual_terminal = vm_lv_pu_trafo * vn_lv_kv (asumiendo vm_lv_pu es referido a vn_lv_kv del trafo)
            # Pandapower res_trafo.vm_hv_pu y vm_lv_pu son referidos a las tensiones nominales de los buses HV y LV a los que se conecta el trafo.
            # Si vn_kv del bus HV es igual a vn_hv_kv del trafo, y vn_kv del bus LV es igual a vn_lv_kv del trafo, entonces:
            bus_hv_idx = self.net.trafo.hv_bus.iloc[0]
            bus_lv_idx = self.net.trafo.lv_bus.iloc[0]
            v_hv_bus_kv = self.net.res_bus.vm_pu.iloc[bus_hv_idx] * self.net.bus.vn_kv.iloc[bus_hv_idx]
            v_lv_bus_kv = self.net.res_bus.vm_pu.iloc[bus_lv_idx] * self.net.bus.vn_kv.iloc[bus_lv_idx]
            
            if v_lv_bus_kv != 0:
                 ratio_from_voltages = v_hv_bus_kv / v_lv_bus_kv
                 #print(f"  Voltaje HV en bus trafo ({self.net.bus.name.iloc[bus_hv_idx]}): {self.net.res_bus.vm_pu.iloc[bus_hv_idx]:.4f} pu ({v_hv_bus_kv:.2f} kV)")
                 #print(f"  Voltaje LV en bus trafo ({self.net.bus.name.iloc[bus_lv_idx]}): {self.net.res_bus.vm_pu.iloc[bus_lv_idx]:.4f} pu ({v_lv_bus_kv:.2f} kV)")
                 #print(f"  Relación de Voltajes en Terminales (V_hv_term / V_lv_term): {ratio_from_voltages:.4f}")

        next_state = self.get_state()
        reward = self.calculate_reward()
        done = self.is_done()
        
        return next_state, reward, done

    def _get_raw_state(self):
        """Método interno para obtener los valores del estado sin correr flujo de potencia adicional."""
        # Asegurarse que los resultados existen (pp.runpp debe haber sido llamado antes)
        if self.net.res_bus is None or 'vm_pu' not in self.net.res_bus:
            # Esto podría pasar si pp.runpp no convergió o no se ha ejecutado
            print("Advertencia: Resultados del flujo de potencia no disponibles en _get_raw_state. Devolviendo estado por defecto.")
            # Estado por defecto con la nueva dimensionalidad (voltaje, p_act, p_react, tap_pos, hora_norm)
            #Número de buses es self.net.bus.shape[0], deberia ser 34
            num_buses = 34
            default_tap_pos = float(self.net.trafo.tap_neutral.iloc[0]) if not self.net.trafo.empty else 16.0
            default_state_values=[0.0]*(3*num_buses)
            default_state_values.extend([default_tap_pos, 0.0])
            return tuple(default_state_values)

            # Usamos tap_neutral como valor por defecto para el tap y 0.0 para la hora normalizada.
            #default_tap_pos = float(self.net.trafo.tap_neutral.iloc[0]) if not self.net.trafo.empty else 16.0
            #return (0.0, 0.0, 0.0, default_tap_pos, 0.0) 
            
        # Parametros que obtenemos del sistema
        # El bus con índice 2 es "Bus 2" en este modelo
        #voltaje = self.net.res_bus.vm_pu.at[6]        #Tensiones en los nodos de la red
        #active_power = self.net.res_bus.p_mw.at[6]    #Potencia activa en los nodos de la red
        #reactive_power = self.net.res_bus.q_mvar.at[6]#Potencia reactiva en los nodos de la red
        all_voltajes=self.net.res_bus.vm_pu.values.tolist()
        all_active_power=self.net.res_bus.p_mw.values.tolist()
        all_reactive_power=self.net.res_bus.q_mvar.values.tolist()

        
        # Información adicional para el estado
        current_tap_pos = float(self.net.trafo.tap_pos.iloc[0])

        # Normalizar la hora actual (self.step_count va de 0 o 1 a MAX_STEPS)
        # Si self.step_count es 0 en reset, y 1-24 durante los pasos:
        # Para el estado inicial (step_count=0), normalized_hour será (0-1)/(24-1) = -1/23 approx -0.043
        # Para el primer paso (step_count=1), normalized_hour será (1-1)/(24-1) = 0
        # Para el último paso (step_count=24), normalized_hour será (24-1)/(24-1) = 1
        normalized_hour = (self.step_count - 1) / (self.MAX_STEPS - 1) if self.MAX_STEPS > 1 else 0.0
        state_list = []
        state_list.extend(all_voltajes) # Correcto: extiende con los elementos de la lista
        state_list.extend(all_active_power) # Correcto: extiende con los elementos de la lista
        state_list.extend(all_reactive_power) # Correcto: extiende con los elementos de la lista
        state_list.extend([current_tap_pos, normalized_hour])
        state=tuple(state_list)
        #state = (voltaje, active_power, reactive_power, current_tap_pos, normalized_hour)
        return state

    #----------- Obtención del estado del sistema 
    def get_state(self): 
        try:
            pp.runpp(self.net)
        except Exception as e:
            print(f"Error al ejecutar el flujo de potencia:{e}")
            # En caso de error, podrías devolver un estado por defecto o propagar el error
            # Devolvemos un estado por defecto con la nueva dimensionalidad
            num_buses=34
            default_tap_pos = float(self.net.trafo.tap_neutral.iloc[0]) if not self.net.trafo.empty else 16.0
            default_state_values=[0.0]*(3*num_buses)
            default_state_values.extend([default_tap_pos, 0.0])
            return tuple(default_state_values)

            #return (0.0, 0.0, 0.0, default_tap_pos, 0.0)
        
        return self._get_raw_state()

    def variables_interes(self):
        Vnodos=self.net.res_bus["vm_pu"]
        Pnodos=self.net.res_bus["p_mw"]
        Qnodos=self.net.res_bus["q_mvar"]

        return Vnodos, Pnodos, Qnodos