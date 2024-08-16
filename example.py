import sklearn
import numpy as np
##############################
# Example Exploration Scheme #
##############################

def explorer(x_t:np.array, u_bounds:dict) -> np.array:
  '''
  Function to collect more data to train state-space model
  Inputs:
  x_t (np.array) - Current state 
  u_bounds (dict) - Bounds on control inputs
  
  Output:
  u_plus - Next control input
  '''
  u_lower = u_bounds['low']
  u_upper = u_bounds['high']
  pass
  # return u_plus



########################
#Example Model Training#
########################

def model_trainer(data):
  '''
  Train a data-driven model (f in the line below) 
  x_plus = f(x_current, u_current)

  Input
  Data - Training data including the timeseries provided and those gathered by your exploration function.

  Output
  Trained_model - trained model (ready to use in your controller!)
  '''
  # TODO: add time assertion
  pass
  # return trained_model


############
# Optimiser#
############

def controller(x, f, sp):
  '''
  DD-MPC Controller to find best next control for the current state and setpoint.
  
  Inputs:
  x - Current state
  f - Your trained data-driven model
  sp - Current setpoint

  Outputs:
  u_plus - Next control

  '''
  # return u