from datetime import datetime,timezone
import do_mpc
from casadi import vertcat,sin,cos,tanh
from CasAdi_Node import CasAdiMLP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegWriter
import os,time,gc

## Algebraic Equation parameters
m_ball = 0.1 ##kg
mu_ball = 1e-3 ##friction coefficient for ball rolling on beam
g = 9.81
L_beam = 1.0
J_beam = 0.5
k_servo = 100.0
b_beam = 10.0
eps = 1e-4


def rk4_step(fnc,state,u,dt=0.01):
    k1 = fnc(state,u)
    k2 = fnc(state + dt/2*k1,u)
    k3 = fnc(state + dt/2*k2,u)
    k4 = fnc(state + dt*k3,u)
    X_next = state + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return X_next

def plant_rhs(X,u):
    x = X[0]
    v = X[1]
    theta = X[2]
    omega= X[3]

    omega_dot_expr = (
    -b_beam*omega
    - k_servo*(theta-u)
    - m_ball*g*x*cos(theta)
    - 2*m_ball*v*omega*x
)/(J_beam + m_ball*x**2)

    v_dot_expr = (
        -g*sin(theta)
        + x*omega**2
        - mu_ball*tanh(v/eps) * (
            g*cos(theta)
            + x*omega_dot_expr
            + 2*v*omega
        )
    )

    return vertcat(v,v_dot_expr,omega,omega_dot_expr)

plt.rcParams['axes.grid'] = True
os.makedirs("animations",exist_ok=True)
os.makedirs("logs",exist_ok=True)

# Initial Conditions (same for both simulations)
x0 = 0.3
sp = 0.6
v0,theta0,omega0 = (0.0 for _ in range(3))
error0 = float(x0-sp)
cluster = 0 if error0 > 0 else 1

print(f"Predicted Cluster: {cluster}")

beam_state = np.array([x0,v0,theta0,omega0],dtype=np.float64).reshape(-1,1)
neural_state = np.array(
    [x0,v0,theta0,omega0],
    dtype = np.float64
).reshape(-1,1)

# Simulation Parameters
model_type = "discrete"
T_total = 5 #secs
t_step = 0.01 #sec

N = int(T_total/t_step)
n_step_predicted = 100

mpc_setup = {
    "n_horizon":n_step_predicted,
    "t_step":t_step,
}

running_weights = np.diag([175,200,100,150])
terminal_weights = running_weights*5

# Defining models
neural_model = do_mpc.model.Model(model_type)
algebraic_model = do_mpc.model.Model(model_type)

## 1. Neural Model States
x = neural_model.set_variable(
    var_type="_x",
    var_name="x"
)

v = neural_model.set_variable(
    var_type="_x",
    var_name="v"
)

theta = neural_model.set_variable(
    var_type="_x",
    var_name="theta"
)

omega = neural_model.set_variable(
    var_type="_x",
    var_name="omega"
)

theta_command = neural_model.set_variable(
    var_type="_u",
    var_name="theta_command"
)

NODE = CasAdiMLP(cluster=cluster)

#RK4
X_next = rk4_step(NODE,vertcat(x,v,theta,omega),theta_command,dt=t_step)

neural_model.set_rhs("x",X_next[0])
neural_model.set_rhs("v",X_next[1])
neural_model.set_rhs("theta",X_next[2])
neural_model.set_rhs("omega",X_next[3])

neural_model.setup()

## 2. Algebraic Model States
x_p = algebraic_model.set_variable(
    var_type="_x",
    var_name="x"
)

v_p = algebraic_model.set_variable(
    var_type="_x",
    var_name="v"
)

theta_p = algebraic_model.set_variable(
    var_type="_x",
    var_name="theta"
)

omega_p = algebraic_model.set_variable(
    var_type="_x",
    var_name="omega"
)

theta_command_p = algebraic_model.set_variable(
    var_type="_u",
    var_name="theta_command"
)

X_next_analytical = rk4_step(plant_rhs,vertcat(x_p,v_p,theta_p,omega_p),theta_command_p,dt=t_step)

algebraic_model.set_rhs("x",X_next_analytical[0])
algebraic_model.set_rhs("v",X_next_analytical[1])
algebraic_model.set_rhs("theta",X_next_analytical[2])
algebraic_model.set_rhs("omega",X_next_analytical[3])
algebraic_model.setup()

# Defining the controller
mpc = do_mpc.controller.MPC(neural_model)
mpc.set_param(**mpc_setup)

## Defining objective function
X = vertcat(x,v,theta,omega)
reference_Vector = vertcat(sp,0.0,0.0,0.0)
Q,P = running_weights,terminal_weights

mterm = (X-reference_Vector).T @ P @ (X-reference_Vector)
lterm = (X-reference_Vector).T @ Q @ (X-reference_Vector) + 1e-4*theta_command**2

mpc.set_objective(
    mterm=mterm,
    lterm=lterm
)

mpc.set_rterm(theta_command = 0.1) #for change in control input

mpc.bounds["lower","_u","theta_command"] = -0.3
mpc.bounds["upper","_u","theta_command"] = 0.3
mpc.bounds["upper","_x","x"] = 0.85
mpc.bounds["lower","_x","x"] = 0.05

mpc.setup()

# Defining the simulators
algebraic_simulator = do_mpc.simulator.Simulator(algebraic_model)

algebraic_simulator.set_param(t_step=t_step)

algebraic_simulator.setup()

algebraic_simulator.x0 = beam_state
mpc.x0 = neural_state
mpc.set_initial_guess()

# Data collection
beam_history = np.zeros((N,4))
control_history = np.zeros((N,1))


start = time.perf_counter()
for i in range(N):


    if i == 0:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

        with open(f"logs/experiment_{ts}.txt","w") as f:
            f.write("*"*20)
            f.write("\n")
            f.write("Starting the experiment with following configuration: ")
            f.write(f"""
x0: {x0},
sp: {sp},
v0: {v0},
theat0: {theta0},
omega0: {omega0},
t_step: {t_step},
total_time: {T_total},
N: {N},
n_steps_predicted: {n_step_predicted},
running_weights: {running_weights}
Remarks: Using discrete models with Rk4.
""")
            f.write("*"*20)
            f.write("\n")
        

    if i%10 == 0:
        with open(f"logs/experiment_{ts}.txt","a") as f:
            f.write(f"At {i}th step in simulation, ")
            f.write(f"Time taken {time.perf_counter() -start} seconds, ")
            f.write("\n")
    
    measured_state = beam_state.copy()
    neural_state = measured_state.copy() 
    algebraic_simulator.x0 = measured_state
    mpc.x0 = neural_state
    u = mpc.make_step(neural_state)

    beam_state = algebraic_simulator.make_step(u)

    beam_history[i] = np.array(beam_state).flatten()
    control_history[i] = np.array(u).flatten()



end = time.perf_counter()
print(f"Control problem simulated in {end-start:.3f} seconds.")

time = np.arange(N)*t_step

beam_position = beam_history[:,0] 

fig,ax = plt.subplots(5,1,figsize=(12,10),sharex=True)

ax[0].plot(time,beam_position,label="Analytical plant")
ax[0].axhline(sp,color='r',ls=':')
ax[0].set_ylabel("x [m]")
ax[0].legend()

ax[1].plot(time,beam_history[:,1],label="Plant")
ax[1].axhline(0.0,color='r',ls=":")
ax[1].set_ylabel("Velocity [m/s]")

ax[2].plot(time,beam_history[:,2])
ax[2].axhline(0.0,color='r',ls=":")
ax[2].set_ylabel(r"$\theta$ [rad]")

ax[3].plot(time,beam_history[:,3])
ax[3].axhline(0.0,color='r',ls=":")
ax[3].set_ylabel(r"$\omega$ [rad/s]")

ax[4].plot(time,control_history)
ax[4].set_ylabel(r"$\theta_c$ [rad]")
ax[4].axhline(0.0,color='k',ls=":")
ax[4].set_xlabel("Time [s]")

plt.tight_layout()
plt.show()
plt.savefig(f"animations/mpc_with_node_{n_step_predicted}_{x0}_{sp}.jpg")
gc.collect()