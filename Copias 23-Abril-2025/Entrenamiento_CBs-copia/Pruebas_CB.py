
import pandapower as pp
# import pandapower.plotting as pp_plot # No usado en este snippet
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from IEEE_33_Bus_System_CB import IEEE33BusSystem
from Actor_Critico_Buffer_CB import Actor
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
env = IEEE33BusSystem()
env.reset()
#-----------------Variables de interes------------------------------
tensiones_sin_control=[]                                                  
aparente_sin_control=[]                                                   
potencia_sin_control=[]                                                   
reactiva_sin_control=[]     

tensiones_control=[]                                                  
aparente_control=[]                                                   
potencia_control=[]                                                   
reactiva_control=[] 
#-----------------Bases del sistema---------------------------------
v_base_kv = env.net.bus.vn_kv.at[2]
s_base_kvar=12.666
#-----------------Caso base sin control-----------------------------
for hora in range (1,25):
   env.update_loads(hora)
   estado=env.get_state()
   tensiones=env.net.res_bus.vm_pu.at[6]
   potencia=env.net.res_bus.p_mw.at[6]
   reactiva=env.net.res_bus.q_mvar.at[6]
   aparente=((potencia**2+reactiva**2)**0.5)
   factor_potencia=potencia/aparente

   tensiones_sin_control.append(tensiones)
   aparente_sin_control.append(aparente)
   potencia_sin_control.append(potencia)
   reactiva_sin_control.append(reactiva)

reactiva_sin_control=np.array(reactiva_sin_control)
potencia_sin_control=np.array(potencia_sin_control)
tensiones_sin_control=np.array(tensiones_sin_control)
aparente_sin_control=np.array(aparente_sin_control)

#-----------------------------------------------------------------------------------------------
for hora in range (1,25):
   env.update_loads(hora)
   estado=env.get_state()
   if hora==21:
      action=3
   else:
      action=3
   env.step(action)
   tensiones=env.net.res_bus.vm_pu.at[6]
   potencia=env.net.res_bus.p_mw.at[6]
   reactiva=env.net.res_bus.q_mvar.at[6]
   aparente=((potencia**2+reactiva**2)**0.5)
   factor_potencia=potencia/aparente

   tensiones_control.append(tensiones)
   aparente_control.append(aparente)
   potencia_control.append(potencia)
   reactiva_control.append(reactiva)


reactiva_control=np.array(reactiva_control)
potencia_control=np.array(potencia_control)
tensiones_control=np.array(tensiones_control)
aparente_control=np.array(aparente_control)


#plt.plot(tensiones_hora, color='g', label='Tension')
#plt.plot(reactiva_hora, color='r', label='Reactiva')
#plt.plot(potencia_hora, color='b', label='Activa')
plt.plot(tensiones_sin_control, color='r', label='Tensiones sin control')
plt.plot(tensiones_control, color='b', label='Tensiones control')
plt.legend()
plt.xlabel('Hora')
#plt.ylabel('Tension en el nodo 2')
#plt.axhline(0.92, color='b', linestyle='--', label='Límite Superior (1.05 pu)')
#plt.axhline(0.91, color='b', linestyle='--', label='Límite Inferior (0.95 pu)')
plt.grid(True)
plt.show()


