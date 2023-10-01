import numpy as np
import sys
from matplotlib import pyplot as plt
import gymnasium as gym

# The path below needs to point to the folder which is MetaGridEnv/MetaGridEnv
sys.path.append("C:/Users/neyvi/Documents/UNI_AI/envoriments_tom/Environments-main/MetaGridEnv")
sys.path.append("C:/Users/neyvi/Documents/UNI_AI/envoriments_tom/Environments-main/MetaGridEnv/MetaGridEnv")
#C:\Users\neyvi\Documents\UNI_AI\envoriments_tom\Environments-main\MetaGridEnv\MetaGridEnv
import MetaGridEnv
from gym.envs.registration import register 

#  Register only needs to be run once (but everytime the script is run)

register( id="MetaGridEnv/metagrid-v0",
          entry_point="metagrid_gymnasium_wrapper:MetaGridEnv")


#env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], style="grid")
env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[7, 21], style="Tori")