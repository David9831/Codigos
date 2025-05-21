import pandapower as pp
import matplotlib.pyplot as plt
import numpy as np
import os
from IEEE_33_Bus_System_CB import IEEE33BusSystem
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
env = IEEE33BusSystem()
tensiones_hora=[]
estado=env.get_state()
tensiones=env.net.res_bus.vm_pu.at[2]
potencia=env.net.res_line.p_from_mw.values
reactiva=env.net.res_line.q_from_mvar.values
plt.plot(tensiones)
#plt.xlabel('Hora')
#plt.ylabel('Tensión pu')
plt.grid(True)
#plt.axhline(1.05, color='r', linestyle='--', label='Límite Superior (1.05 pu)')
#plt.axhline(0.95, color='r', linestyle='--', label='Límite Inferior (0.95 pu)')
plt.show()