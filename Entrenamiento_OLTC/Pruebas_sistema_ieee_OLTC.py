import pandapower as pp
import pandapower.plotting as pp_plot
import matplotlib.pyplot as plt
import numpy as np
import os
from IEEE_34_Bus_System import IEEE33BusSystem
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
env=IEEE33BusSystem()
hora=3
carga=env.update_loads(hora)
print(carga)
#x=np.arange(0,34)
#limite_superior=1.05
#limite_inferior=0.95

#plt.figure(figsize=(10,6))
#plt.axhline(y=limite_superior, color='r', linestyle='--')
#plt.axhline(y=limite_inferior, color='r', linestyle='--')
#plt.plot(x, Vnodos, marker='o', linestyle='-')
#plt.xlabel("Barra")
#plt.ylabel("Voltaje (Pu)")
#plt.title(f"Perfil de voltaje")
#plt.grid(True)
#plt.tight_layout()
#plt.show()