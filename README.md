**Model Predictive Control (MPC) for SISO and MIMO Systems**

Developed the project to understand Model Predictive Control (MPC) concepts. Two systems are implemented here: a basic single-input-single-output (SISO) system and a more complex multi-input-multi-output (MIMO) system. The simulations were created using Python with the do_mpc library.

**Requirements**

The following Python packages are needed:

numpy

matplotlib

do_mpc

**
SISO Implementation**

"mpc.py" - implements a simple system with one state variable and one control input. The system dynamics are defined as a discrete-time model where the next state depends on the current state and control input. A prediction horizon of 10 steps was chosen for the controller.

Control constraints are set between -1 and 1, which means the controller can't apply input beyond these limits. The cost function minimizes both the state value and the control effort.

After running for 30 time steps, the system's state is driven to zero from its initial value of 2.0, which shows the controller works as expected.

**MIMO Implementation**

"mpc_MIMO.py" - extends to a two-state, two-input system. The states interact with each other through the dynamics matrix, making this more challenging to control. The system matrices are:

A = [[0.9, 0.1], [0.2, 0.8]]
B = [[1.0, 0.0], [0.0, 1.0]]

For this system, control inputs are bounded between -1.5 and 1.5. The initial states are set to [2.0, -3.0], and the controller successfully drives both states to zero.

**Results**

Both implementations show how MPC can drive a system to stability while respecting constraints. The plots generated show:

- How states change over time

- What control inputs are applied at each step

- The SISO system converges smoothly to zero, while the MIMO system shows more complex behavior due to the interaction between states, but still achieves stability.

Notes
These implementations are meant for learning purposes. Real-world systems would likely need more tuning and might have additional constraints or disturbances to handle.

**Future Work**

- Working on to implement nonlinear system dynamics

- Testing different cost functions and constraints

- Comparing MPC with other control strategies
