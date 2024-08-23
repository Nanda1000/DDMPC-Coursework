from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize, Bounds
from sklearn.metrics import mean_squared_error, r2_score
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

##############################
# Example Exploration Scheme #
##############################

def explorer(x_t: np.array, u_bounds: dict, timestep: int) -> np.array:
    '''
    Function to collect more data to train state-space model using step changes every 30 timesteps
    Inputs:
    x_t (np.array) - Current state 
    u_bounds (dict) - Bounds on control inputs
    timestep (int) - Current timestep
    
    Output:
    u_plus - Next control input
    '''
    u_lower = u_bounds['low']
    u_upper = u_bounds['high']

    return np.random.uniform(u_lower, u_upper, size=u_lower.shape)
    

########################
#Example Model Training#
########################

def model_trainer(data, env, time_limit=300):
    data_states, data_controls = data
    
    # Select only states with indices 1 and 8
    selected_states = data_states[:, [1, 8], :]
    
    # Normalize the selected states and controls
    selected_states_norm = (selected_states - env.env_params['o_space']['low'][[1, 8]].reshape(1, -1, 1)) / (env.env_params['o_space']['high'][[1, 8]].reshape(1, -1, 1) - env.env_params['o_space']['low'][[1, 8]].reshape(1, -1, 1)) * 2 - 1
    data_controls_norm = (data_controls - env.env_params['a_space']['low'].reshape(1, -1, 1)) / (env.env_params['a_space']['high'].reshape(1, -1, 1) - env.env_params['a_space']['low'].reshape(1, -1, 1)) * 2 - 1

    # Get the dimensions
    reps, states, n_steps = selected_states_norm.shape
    _, controls, _ = data_controls_norm.shape
    
    # Prepare the data
    X_states = selected_states_norm[:, :, :-1].reshape(-1, states)
    X_controls = data_controls_norm[:, :, :-1].reshape(-1, controls)
    X = np.hstack([X_states, X_controls])
    y = selected_states_norm[:, :, 1:].reshape(-1, states)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.3f}")

    return model




############
# Optimiser#
############

def controller(x, f, sp, env, u_prev):


    """
    Model Predictive Controller
    
    Args:
    x (numpy.array): Current state of the system.
    f (object): Your trained model used for predicting the next state.
    sp (numpy.array): Setpoint for the system.
    env (object): Environment object containing parameters.
    u_prev (numpy.array): Previous control action.
    numpy.array: First control action from MPC optimization.

    Returns:
    optimal_control: First control action from MPC optimization
    """
    horizon = 10 # Control horizon
    x_current = x[1]

    n_controls = env.env_params['a_space']['low'].shape[0]
    u_prev = (u_prev - env.env_params['a_space']['low']) / (env.env_params['a_space']['high'] - env.env_params['a_space']['low']) * 2 - 1
    
    def predict_next_state(current_state, control):
        # Ensure the input is normalized as in the training data
        current_state_norm = (current_state - env.env_params['o_space']['low'][[1,8]]) / (env.env_params['o_space']['high'][[1,8]] - env.env_params['o_space']['low'][[1,8]]) * 2 - 1
        x = np.hstack([current_state_norm, control])
        prediction = f.predict(x.reshape(1, -1)).flatten()
        prediction = (prediction + 1) / 2 * (env.env_params['o_space']['high'][[1,8]] - env.env_params['o_space']['low'][[1,8]]) + env.env_params['o_space']['low'][[1,8]]
        return prediction.flatten()
    
    def objective(u_sequence):
        cost = 0
        x_pred = x_current
        R = 1 # Weight for control effort
        Q =  10 # Weight for state error

        for i in range(horizon):
            # State cost
            error = x_pred - sp
            cost += np.sum((error)**2) * Q

            u_current = u_sequence[i*n_controls:(i+1)*n_controls]
            delta_u = u_current - u_prev
            cost += np.sum(np.square(delta_u)) * R 
            x_pred = predict_next_state(x_pred, u_current)
        
        return cost
    
    u_init = np.ones((horizon, 2))*u_prev
    bounds = []
    bounds = [(-1, 1)] * (horizon * n_controls)

    result = minimize(
        objective,
        u_init.flatten(),
        method='powell',
        bounds=bounds
    )
    optimal_control = result.x[:2]
    optimal_control = (optimal_control + 1) / 2 * (env.env_params['a_space']['high'] - env.env_params['a_space']['low']) + env.env_params['a_space']['low']
    
    return optimal_control