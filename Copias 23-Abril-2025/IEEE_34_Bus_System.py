import pandapower as pp
import numpy as np
import matplotlib.pyplot as plt
import pandapower.plotting as pp_plot
import torch.nn as nn
import torch.nn.functional as F 
import torch

class IEEE33BusSystem:
    def __init__(self):
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
        self.setup_network()
        self.state_dim = self.get_state()
        self.step_count = 0
        self.MAX_STEPS = 24

    def is_done(self):
        # Terminar el episodio si se alcanza el límite de pasos
        if self.step_count == self.MAX_STEPS:
            return True
        return False

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
            vn_lv_kv=12.66,
            pfe_kw=110, 
            i0_percent=0, 
            vk_percent=8, 
            vkr_percent=0.5, 
            tap_side="hv", 
            tap_neutral=16,
            tap_min=0, 
            tap_max=33, 
            tap_step_percent=1.25,             #Revisar el porcentaje de carga por paso del transformador (¿¿¿1.25% no es muy poco???)
            tap_pos=16
        )
        self.action_dim=33      #Cantidad de taps del transformador 
        # Agregar líneas de distribución
        a=0
        b=1
        pp.create_line_from_parameters(self.net, from_bus=b0, to_bus=b1, length_km=1, r_ohm_per_km=0.0922,x_ohm_per_km=0.0470, c_nf_per_km=a, max_i_ka=b)
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
    def update_loads(self,hora):
        for j in range (hora):                                                        
            for i, (p_base, q_base) in enumerate(self.base_loads):
                   self.net.load.at[i, 'p_mw'] = p_base * self.porcentaje_demanda[j]
                   self.net.load.at[i, 'q_mvar'] = q_base * self.porcentaje_demanda[j]
        carga_hora=self.net.load
        return carga_hora
        