import numpy as np
import matplotlib.pyplot as plt
import do_mpc

# ---------------------------
# Step 1: Define the Model
# ---------------------------
model_type = 'discrete'
model = do_mpc.model.Model(model_type)

# Define state and control variables with 2D shapes (1,1)
x1 = model.set_variable(var_type='_x', var_name='x1', shape=(1,1))
u1 = model.set_variable(var_type='_u', var_name='u1', shape=(1,1))

# Define system dynamics: x_next = A*x + B*u
A = np.array([[1.0]])
B = np.array([[1.0]])
model.set_rhs('x1', A[0, 0] * x1 + B[0, 0] * u1)
model.setup()



# ---------------------------
# Step 2: Set Up the MPC Controller
# ---------------------------
mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': 10,
    't_step': 1.0,
    'state_discretization': 'discrete',
    'store_full_solution': True,
    'nlpsol_opts': {'ipopt.print_level': 0,
                    'print_time': 0,
                    'ipopt.sb': 'yes'}
}
mpc.set_param(**setup_mpc)

mterm = x1**2        # terminal cost
lterm = x1**2 + u1**2  # stage cost
mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(u1=0.1)

# Set control bounds
mpc.bounds['lower', '_u', 'u1'] = -1.0
mpc.bounds['upper', '_u', 'u1'] = 1.0
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
# You can use either Option 1 (define initial state as 2D) or Option 2.
# Here we use Option 2:
x0 = np.array([2.0])  # initial state (shape (1,))
simulator.x0 = x0
mpc.x0 = x0

# Lists to store the simulation results as scalars
x_history = [np.array(x0).flatten()[0]]
u_history = []

n_steps = 30

for k in range(n_steps):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    # Convert each state and control to a scalar consistently
    x_history.append(np.array(x0).flatten()[0])
    u_history.append(np.array(u0).flatten()[0])

# Convert lists to NumPy arrays for plotting
x_history = np.array(x_history)
u_history = np.array(u_history)

# ---------------------------
# Step 5: Visualize the Results
# ---------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_history, marker='o')
plt.title('State Evolution (x1)')
plt.xlabel('Time Step')
plt.ylabel('State x1')

plt.subplot(1, 2, 2)
plt.step(range(len(u_history)), u_history, where='post', marker='o')
plt.title('Control Input Evolution (u1)')
plt.xlabel('Time Step')
plt.ylabel('Control Input u1')

plt.tight_layout()
plt.show()
