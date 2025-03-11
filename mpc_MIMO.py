import numpy as np
import matplotlib.pyplot as plt
import do_mpc

# ---------------------------
# Step 1: Define the MIMO Model
# ---------------------------
model_type = 'discrete'
model = do_mpc.model.Model(model_type)

# Define state and control variables
x1 = model.set_variable(var_type='_x', var_name='x1', shape=(1, 1))
x2 = model.set_variable(var_type='_x', var_name='x2', shape=(1, 1))
u1 = model.set_variable(var_type='_u', var_name='u1', shape=(1, 1))
u2 = model.set_variable(var_type='_u', var_name='u2', shape=(1, 1))

# Define system dynamics: x_next = A*x + B*u
A = np.array([[0.9, 0.1], [0.2, 0.8]])
B = np.array([[1.0, 0.0], [0.0, 1.0]])

model.set_rhs('x1', A[0, 0] * x1 + A[0, 1] * x2 + B[0, 0] * u1)
model.set_rhs('x2', A[1, 0] * x1 + A[1, 1] * x2 + B[1, 1] * u2)

model.setup()

# ---------------------------
# Step 2: Set Up the MPC Controller
# ---------------------------
mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': 10,                    # Prediction horizon
    't_step': 1.0,                      # Time step
    'state_discretization': 'discrete',
    'store_full_solution': True,
    'nlpsol_opts': {'ipopt.print_level': 0,
                    'print_time': 0,
                    'ipopt.sb': 'yes'}
}
mpc.set_param(**setup_mpc)

# Define cost function
mterm = x1**2 + x2**2       # Terminal cost
w_u = 1.5 # Weight for control input
lterm = x1**2 + x2**2 + w_u * (u1**2 + u2**2)  # Stage cost

mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(u1=0.05, u2=0.05)  # Regularization terms for both controls

# Set control bounds for u1 and u2
mpc.bounds['lower', '_u', 'u1'] = -1.5
mpc.bounds['upper', '_u', 'u1'] = 1.5
mpc.bounds['lower', '_u', 'u2'] = -1.5
mpc.bounds['upper', '_u', 'u2'] = 1.5

mpc.setup()
mpc.set_initial_guess()

# ---------------------------
# Step 3: Set Up the Simulator
# ---------------------------
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=setup_mpc['t_step'])
simulator.setup()

# ---------------------------
# Step 4: Run the Closed-Loop Simulation
# ---------------------------
x0 = np.array([2.0, -3.0])  # Initial states for x1 and x2
simulator.x0 = x0
mpc.x0 = x0

# Lists to store simulation results for both states and controls
x_history = [x0]
u_history = []

n_steps = 30

for k in range(n_steps):
    u0 = mpc.make_step(x_history[-1])   # Compute optimal control input
    next_state = simulator.make_step(u0)   # Update state using simulator dynamics
    # Store results
    x_history.append(np.array(next_state).flatten())
    u_history.append(np.array(u0).flatten())

# Convert lists to NumPy arrays for plotting
x_history = np.array(x_history)
u_history = np.array(u_history)

# ---------------------------
# Step 5: Visualize the Results
# ---------------------------
plt.figure(figsize=(12, 8))

# State evolution plot (x1 and x2)
plt.subplot(2, 2, 1)
plt.plot(x_history[:, 0], marker='o')
plt.title('State Evolution (x1)')
plt.xlabel('Time Step')
plt.ylabel('State x1')

plt.subplot(2, 2, 3)
plt.plot(x_history[:, 1], marker='o')
plt.title('State Evolution (x2)')
plt.xlabel('Time Step')
plt.ylabel('State x2')

# Control input evolution plot (u1 and u2)
plt.subplot(2, 2, 2)
plt.step(range(len(u_history)), u_history[:, 0], where='post', marker='o')
plt.title('Control Input Evolution (u1)')
plt.xlabel('Time Step')
plt.ylabel('Control Input u1')

plt.subplot(2, 2, 4)
plt.step(range(len(u_history)), u_history[:, 1], where='post', marker='o')
plt.title('Control Input Evolution (u2)')
plt.xlabel('Time Step')
plt.ylabel('Control Input u2')

plt.tight_layout()
plt.show()
