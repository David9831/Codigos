import pandapower as pp
import pandapower.plotting as pp_plot
import matplotlib.pyplot as plt
import numpy as np
import os
from IEEE_34_Bus_System_OLTC import IEEE33BusSystem
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
env=IEEE33BusSystem()
action=1
env.step(action)
Vnodo=env.net.res_bus['vm_pu']
print(Vnodo)

