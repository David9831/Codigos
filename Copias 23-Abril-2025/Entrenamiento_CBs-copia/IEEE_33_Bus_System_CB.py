import pandapower as pp
import numpy as np
import matplotlib.pyplot as plt
import pandapower.plotting as pp_plot


class IEEE33BusSystem:
    def __init__(self):
        self.net = pp.create_empty_network(f_hz=60,sn_mva=1)
        # Base loads for 32 buses (from bus 1 to 32)
        self.base_loads = [
            (0.1, 0.06), (0.09, 0.04), (0.12, 0.08), (0.06, 0.03), (0.06, 0.02),
            (0.2, 0.1), (0.2, 0.1), (0.06, 0.02), (0.06, 0.02), (0.045, 0.03),
            (0.06, 0.035), (0.06, 0.035), (0.12, 0.08), (0.06, 0.01), (0.06, 0.02),
            (0.06, 0.02), (0.09, 0.04), (0.09, 0.04), (0.09, 0.04), (0.09, 0.04),
            (0.09, 0.04), (0.09, 0.05), (0.42, 0.2), (0.42, 0.2), (0.06, 0.025),
            (0.06, 0.025), (0.06, 0.02), (0.12, 0.07), (0.2, 0.6), (0.15, 0.07),
            (0.21, 0.1), (0.06, 0.04)
        ]
        self.porcentaje_demanda = [
            0.28, 0.25, 0.26, 0.25, 0.23, 0.44, 0.69, 0.44, 0.44, 0.36, 0.38, 0.51,
            0.4, 0.38, 0.37, 0.41, 0.49, 0.34, 0.61, 0.81, 1.0, 0.86, 0.69, 0.36
        ] # Hourly load profile factor (for hours 1 to 24)

        # RL Environment Configuration
        self.STATE_DIM = 4 # Voltage, Active Power, Reactive Power (at target bus), Capacitor Step
        self.ACTION_DIM = 4 # Number of discrete capacitor steps (0, 1, 2, 3)
        self.MAX_STEPS = 24 # Number of hours in a day
        self.TARGET_BUS_INDICES = [6] # Bus indices to include in the state and reward calculation
        self.CONTROLLED_SHUNT_INDEX = 0 # Index of the shunt element controlled by the agent

        self.setup_network() 
        self.step_count = 0 # Current step within the episode (hour of the day)

    def is_done(self):
        # Terminar el episodio si se alcanza el límite de pasos
        if self.step_count == self.MAX_STEPS:
            return True
        return False

    def setup_network(self):
        # Create buses (Bus 0 is the slack bus connected via transformer)
        Vnhkv=110 # High voltage side of the transformer
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
            vn_lv_kv=12.66,
            pfe_kw=110, 
            i0_percent=0, 
            vk_percent=8, 
            vkr_percent=0.5, 
            tap_side="hv", 
            tap_neutral=16,
            tap_min=0, 
            tap_max=33, 
            tap_step_percent=1.25, 
            tap_pos=16
        )
        # Add controllable shunt capacitor at bus 6
        # max_step = 3 means steps 0, 1, 2, 3 are possible (4 discrete actions)
        # q_mvar is the reactive power per step. -0.15 Mvar per step (capacitive)
        pp.create_shunt(self.net, bus=6, q_mvar=-0.15, step=0, max_step=3, name="Controllable Capacitor at Bus 6")

        # Agregar líneas de distribución
        a=0
        b=1
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

#--------------Actualización de las cargas dependiendo de la demanda horaria-------------------------
    def update_loads(self, hora):
        """Updates loads based on the hourly demand factor."""
        # 'hora' is 1-indexed (1 to 24), 'porcentaje_demanda' is 0-indexed (0 to 23)
        demand_factor = self.porcentaje_demanda[hora - 1]
        for i, (p_base, q_base) in enumerate(self.base_loads):
            self.net.load.at[i, 'p_mw'] = p_base * demand_factor
            self.net.load.at[i, 'q_mvar'] = q_base * demand_factor

#--------------Resetear el sistema de distribucion con las cargas iniciales-------------------------
    def reset(self):
        """Resets the network loads to base values and resets step count."""
        for i, (p_base, q_base) in enumerate(self.base_loads):
            self.net.load.at[i, 'q_mvar'] = q_base
            carga0=self.net.load
        return carga0
#--------------Función de recompensa-------------------------------------------------------
    def calculate_reward(self, previous_step, current_step):
        # --- Penalización por Voltaje ---
        # Asegúrate de que res_bus no esté vacío y tenga 'vm_pu'
        if self.net.res_bus.empty or 'vm_pu' not in self.net.res_bus.columns:
            print("Warning: res_bus empty or missing 'vm_pu' when calculating reward.")
            penalizacion_voltaje = 1000 # Penalización alta si no hay resultados
        else:
            voltajes = self.net.res_bus.loc[self.TARGET_BUS_INDICES, 'vm_pu'].values
            # Penalize voltages outside the [0.95, 1.05] pu range
            penalizacion_voltaje = np.sum(np.maximum(0, 0.95 - voltajes)**2 +
                                          np.maximum(0, voltajes - 1.05)**2)

        # Aquí mantenemos tu cálculo original (penaliza toda Q en buses)
        if self.net.res_bus.empty or 'q_mvar' not in self.net.res_bus.columns:
             penalizacion_reactiva = 1000 # Penalización alta
        else:
            potencia_reactiva_buses = np.abs(self.net.res_bus.loc[self.TARGET_BUS_INDICES, 'q_mvar'].values)
            penalizacion_reactiva = np.sum(potencia_reactiva_buses)
        # --- Penalización por Acción del Capacitor ---
        step_change = abs(current_step - previous_step)
        penalizacion_accion = step_change ** 2

        w1 = 10.0   # Peso para desviación de voltaje
        w2 = 0.1    # Peso para potencia reactiva (reducido si el foco es voltaje)
        w3 = 5.0    # Peso para cambio de step del capacitor

        normalization_factor = 100 # Ajusta este factor según sea necesario
        reward = -(w1 * penalizacion_voltaje + w2 * penalizacion_reactiva + w3 * penalizacion_accion) / normalization_factor
        return reward
#--------------Definición de la acción del actor---------------------------------------------------   
    def step(self, action):
        """
        Aplica la acción (nuevo step del capacitor), ejecuta el flujo de potencia,
        calcula la recompensa y obtiene el siguiente estado.

        Args:
            action (int): El nuevo paso deseado para el capacitor (debe ser un entero).

        Returns:
            tuple: (next_state, reward, done)
        """
        # Asegurar que la acción es un entero
        action = int(action)
        # Get max step for the controlled shunt
        max_step = int(self.net.shunt.at[self.CONTROLLED_SHUNT_INDEX, 'max_step'])
        # Clip action to be within the valid range [0, max_step]
        action = np.clip(action, 0, max_step)

        # 1. Obtener el estado del capacitor ANTES de aplicar la acción
        try:
            previous_step = int(self.net.shunt.at[self.CONTROLLED_SHUNT_INDEX, 'step'])
        except (KeyError, IndexError):
            print(f"Error: Could not get previous step for shunt {self.CONTROLLED_SHUNT_INDEX}.") # Mantener esta línea
            previous_step = 0 # Asumir 0 si no se puede obtener
        except Exception as e:
             print(f"Error inesperado obteniendo previous_step: {e}")
             # Eliminar el bloque if/else problemático que estaba aquí y causaba el IndentationError.
             # La acción ya se valida con np.clip anteriormente.
             # La aplicación de la acción se hace en el siguiente bloque try.
             previous_step = 0 # Asumir 0 si hay un error inesperado al obtener el paso previo

        # 2. Apply the action (change the capacitor step)
        try:
            self.net.shunt.at[self.CONTROLLED_SHUNT_INDEX, 'step'] = action
        except (KeyError, IndexError):
             print(f"Error: Could not apply action {action} to shunt {self.CONTROLLED_SHUNT_INDEX}.")
             # Si la aplicación de la acción falla, el estado del shunt no cambia.
             # La recompensa se calculará con el estado real del shunt. No se reasigna 'action'.
        except Exception as e: # Capturar otras posibles excepciones al aplicar la acción
            print(f"Error inesperado aplicando acción {action} al shunt {self.CONTROLLED_SHUNT_INDEX}: {e}")

        # 3. Ejecutar el flujo de potencia
        try:
            pp.runpp(self.net, algorithm='nr', calculate_voltage_angles=True)
            converged = True
        except pp.LoadflowNotConverged as e:
            print(f"Warning: Power flow did not converge at step {self.step_count}. Action={action}. Error: {e}")
            converged = False
            # ¿Qué hacer si no converge?
            # Opción 1: Recompensa muy negativa
            # Opción 2: Devolver estado anterior (o de ceros) y recompensa negativa
            # Opción 3: Terminar el episodio (done=True)
            # Returning zero state and negative reward
            reward = -10.0 # Recompensa muy negativa
            next_state = self.get_state() # Intentar obtener estado (puede ser de ceros si get_state lo maneja)
            done = self.is_done() # Comprobar si se alcanzó el límite de pasos
            self.step_count += 1 # Incrementar contador incluso si no converge
            return next_state, reward, done
        except Exception as e:
            print(f"Error inesperado en pp.runpp: {e}")
            # Handle other power flow errors
            reward = -10.0
            next_state = self.get_state()
            done = self.is_done()
            self.step_count += 1
            return next_state, reward, done

        # 4. Obtener el siguiente estado
        # Note: get_state does NOT run pp.runpp, it just reads results.
        # pp.runpp was already called above.
        next_state = self.get_state()


        # 5. Calcular la recompensa usando el step anterior y el actual (acción)
        # Asegurarse de que 'action' es el paso que realmente se aplicó
        current_step_applied = int(self.net.shunt.at[self.CONTROLLED_SHUNT_INDEX, 'step'])
        reward = self.calculate_reward(previous_step, current_step_applied)

        # 6. Incrementar contador y verificar si el episodio terminó
        self.step_count += 1
        done = self.is_done()

        return next_state, reward, done
#-----------Obtención del estado del sistema-------------------------------------------------------------- 
    def get_state(self):
        """
        Gets the current state of the system.
        Assumes pp.runpp has been called just before this method.
        State includes voltage, active power, reactive power at target buses,
        and the current capacitor step.
        """
        # Check if results are available after runpp (called in step)
        if self.net.res_bus.empty:
             print("Error: res_bus table is empty. Power flow might not have run or converged.")
             print(f"Warning: Returning zero state ({self.STATE_DIM}) due to empty res_bus.")
             return np.zeros(self.STATE_DIM, dtype=np.float32)
             
        # --- Corrected data extraction using .loc ---
        # Using .loc on the DataFrame is generally robust
        try:
            voltaje = self.net.res_bus.loc[self.TARGET_BUS_INDICES, 'vm_pu'].values
            active_power = self.net.res_bus.loc[self.TARGET_BUS_INDICES, 'p_mw'].values
            reactive_power = self.net.res_bus.loc[self.TARGET_BUS_INDICES, 'q_mvar'].values
        except KeyError as e:
            print(f"Error: One of the target bus indices {self.TARGET_BUS_INDICES} not found in res_bus. Available indices: {self.net.res_bus.index.tolist()}. Error: {e}")
            print(f"Warning: Returning zero state ({self.STATE_DIM}) due to KeyError.")
            return np.zeros(self.STATE_DIM, dtype=np.float32)
        except Exception as e:
            print(f"Unexpected error extracting data from res_bus: {e}")
            raise

        # --- Concatenate into a single state vector ---
        # Ensure all parts have the expected length (1 for each target bus metric + 1 for capacitor step)
        # Assuming TARGET_BUS_INDICES has length 1
        if not (len(voltaje) == len(self.TARGET_BUS_INDICES) and
                len(active_power) == len(self.TARGET_BUS_INDICES) and
                len(reactive_power) == len(self.TARGET_BUS_INDICES)):
             print(f"Error: Unexpected length of extracted vectors. V:{len(voltaje)}, P:{len(active_power)}, Q:{len(reactive_power)}. Expected: {len(self.TARGET_BUS_INDICES)}")
             print(f"Warning: Returning zero state ({self.STATE_DIM}) due to incorrect vector length.")
             return np.zeros(self.STATE_DIM, dtype=np.float32)

        try:
            capacitor_step = self.net.shunt.at[self.CONTROLLED_SHUNT_INDEX, 'step']
            # Convertir a un array numpy para concatenación
            capacitor_step_array = np.array([capacitor_step], dtype=np.float32)
        except IndexError:
            print("Error: No se encontró el shunt (capacitor) en el índice 0.")
            print(f"Warning: Returning zero state ({self.STATE_DIM}) due to missing shunt.")
            return np.zeros(self.STATE_DIM, dtype=np.float32)
        except KeyError:
             print("Error: La columna 'step' no existe en el DataFrame de shunts.")
             print(f"Warning: Returning zero state ({self.STATE_DIM}) due to missing 'step' column.")
             return np.zeros(self.STATE_DIM, dtype=np.float32)
        except Exception as e:
            print(f"Error inesperado al obtener el estado del capacitor: {e}")
            raise

        state = np.concatenate((voltaje, active_power, reactive_power, capacitor_step_array), axis=0) # Concatenate along axis 0
        # Ensure the final state has the correct dimension
        if state.shape[0] != self.STATE_DIM:
             print(f"Error: Final state dimension ({state.shape[0]}) does not match self.STATE_DIM ({self.STATE_DIM}).")
             # This shouldn't happen if concatenation is correct, but good sanity check
             print(f"Warning: Returning zero state ({self.STATE_DIM}) due to incorrect final dimension.")
             return np.zeros(self.STATE_DIM, dtype=np.float32)
        return state.astype(np.float32) # Return a single NumPy array of shape (STATE_DIM,)
    
    def variables_interes(self):
        #Vnodo=self.net.res_bus.vm_pu.at[7]
        #Pnodos=self.net.res_bus.p_mw.at[7]
        #Pnodos=self.net.res_bus["p_mw"]
        #Qnodos=self.net.res_bus["q_mvar"]
        #Qcapacitor=self.net.res_shunt["q_mvar"]
        Pnodos=self.net.res_bus["p_mw"]