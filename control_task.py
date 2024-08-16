from pcgym import make_env # Give it a star on Github :)
import numpy as np
from utils import reward_fn, rollout # functions to make the code below more readable 
from example import explorer, model_trainer, controller # Your algorithms 


#######################################
#  Multistage Extraction Column model #
#######################################
T = 26
nsteps = 120
SP = {
      'X5': [0.3 for i in range(int(nsteps/4))] + [0.4 for i in range(int(nsteps/2))]+ [0.3 for i in range(int(nsteps/4))],
      'Y1': [0.3 for i in range(int(nsteps/2))] + [0.35 for i in range(int(nsteps/2))]
  }

action_space = {
    'low': np.array([5,10]),
    'high':np.array([500,1000])
}

observation_space = {
    'low' : np.array([0]*10+[0.3]),
    'high' : np.array([1]*10+[0.4])  
}

env_params_ms = {
    'N': nsteps,
    'tsim':T,
    'SP':SP,
    'o_space' : observation_space,
    'a_space' : action_space,
    'x0': np.array([0.55, 0.3, 0.45, 0.25, 0.4, 0.20, 0.35, 0.15, 0.25, 0.1,0.3]),
    'model': 'multistage_extraction', 
    'noise':True, #Add noise to the states
    'noise_percentage':0.01,
    'integration_method': 'casadi',
    'custom_reward': reward_fn
}

env = make_env(env_params_ms)

#########################
#Import pre-defined data#
#########################




###################
#Exploration Phase#
###################

N_sim = 50 # Budget for simulations to gather extra data 
data = []
for i in range(N_sim):
    x_log, u_log = rollout(env=env, explore=True, explorer=explorer)
    data.append(x_log, u_log)


#################
# Training Phase#
#################


model = model_trainer(data)

###############
#Control Phase#
###############

N_reps = 10 # Repetitions of simulation with data-driven controller

for i in range(N_sim):
    # TODO: Add changing setpoints
    x_log, u_log = rollout(env=env, explore=False, explorer=controller, model=model)

