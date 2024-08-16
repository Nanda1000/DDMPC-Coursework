import numpy as np
from typing import Callable
def reward_fn(self, x:np.array, u:np.array,con: bool) -> float:
    Sp_i = 0
    cost = 0 
    R = 4
    for k in self.env_params["SP"]:
        i = self.model.info()["states"].index(k)
        SP = self.SP[k]

        o_space_low = self.env_params["o_space"]["low"][i] 
        o_space_high = self.env_params["o_space"]["high"][i] 

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        r_scale = self.env_params.get("r_scale", {})

        cost += (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2) * r_scale.get(k, 1)

        Sp_i += 1
    u_normalized = (u - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )

    # Add the control cost
    cost += R * u_normalized**2
    r = -cost
    return r



def rollout(env:Callable,  explore:bool, explorer=None, controller = None, model=None) -> tuple[np.array, np.array]:
    nsteps = env.N
    sp = env.SP
    x_log = np.empty(env.x0.shape - len(sp), env.N)
    u_log = np.empty(env.Nu.shape, env.N)


    x, _ = env.reset()
    for i in range(nsteps - 1):
        if explore:
            u = explorer(x, env.env_params['a_space'])
        else:
            u = controller(x, model, sp[:,i])
        
        x, _, _, _,  = env.step(u)
        
        x_log[:, i] = x[:len(sp)]
        u_log[:, i] = u
    
    return x_log, u_log
    