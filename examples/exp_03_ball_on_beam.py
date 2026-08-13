import time
from casadi import sign,sin,cos,vertcat,tanh
import do_mpc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import os

plt.rcParams['axes.grid'] = True
os.makedirs("animations",exist_ok=True)

# Integrate neural ode in this 

##Parameters 
model_type = "continuous"
m_ball = 0.1 ##kg
mu_ball = 1e-3 ##friction coefficient for ball rolling on beam
g = 9.81
L_beam = 1.0
J_beam = 0.5
k_servo = 100.0
b_beam = 10.0
t_step = 0.02
eps = 1e-4
mpc_setup = {
    "n_horizon":30,
    "t_step":t_step,
    "store_full_solution":True
}

## Initial condistions
x_initial = 0.1
x0 = np.array([x_initial,0.0,0.0,0.0]).reshape(-1,1)
sp = 0.9
reference_vector = np.array([sp,0.0,0.0,0.0]).reshape(-1,1)
running_weights = np.diag([150,50,10,5])
terminal_weights = running_weights*5

model = do_mpc.model.Model(model_type)

x = model.set_variable(
    var_type = "_x",
    var_name = "x"
)
v= model.set_variable(
    var_type = "_x",
    var_name = "v"
)
theta = model.set_variable(
    var_type ="_x",
    var_name="theta"
)
omega = model.set_variable(
    var_type = "_x",
    var_name = "omega"
)
theta_command = model.set_variable(
    var_type = "_u",
    var_name= "theta_command"
)


omega_dot_expr = (
    -b_beam*omega
    - k_servo*(theta-theta_command)
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


model.set_rhs("x",v)
model.set_rhs("v",v_dot_expr)
model.set_rhs("theta",omega)
model.set_rhs(
    "omega",
    omega_dot_expr
)
model.setup() 

## Defining the controller
mpc = do_mpc.controller.MPC(model)
mpc.set_param(**mpc_setup)

# Defining the loss function

X = vertcat(x,v,theta,omega)
X_ref = reference_vector
Q = running_weights
P = terminal_weights

dx = X - X_ref

mterm = dx.T @ P @ dx
lterm = dx.T @ Q @ dx + theta_command * 0.1 * theta_command

mpc.set_objective(
    mterm=mterm,
    lterm=lterm
)

mpc.set_rterm(theta_command=1.0)

mpc.bounds["lower","_u","theta_command"] = -0.3
mpc.bounds["upper","_u","theta_command"] = 0.3
mpc.bounds["upper","_x","x"] = 1.0
mpc.bounds["lower","_x","x"] = 0.0

mpc.setup()

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = t_step)
simulator.setup()

simulator.x0 = x0
mpc.x0 = x0

mpc.set_initial_guess()

start = time.perf_counter()
for _ in range(500):
    u = mpc.make_step(x0)
    x0 = simulator.make_step(u)
end = time.perf_counter()

print(f"Finished the control problem in {end-start:.3f} seconds.")

graphics = do_mpc.graphics.Graphics(mpc.data)

fig,ax = plt.subplots(5,1,figsize=(12,10),sharex=True,constrained_layout = True)

graphics.add_line("_x","x",axis=ax[0])
graphics.add_line("_x","v",axis=ax[1])
graphics.add_line("_x","theta",axis=ax[2])
graphics.add_line("_x","omega",axis=ax[3])
graphics.add_line("_u","theta_command",axis=ax[-1])

ax[0].axhline(sp,color='r',ls='--')
ax[1].axhline(0,color='r',ls='--')
ax[2].axhline(0,color='r',ls='--')
ax[3].axhline(0,color='r',ls='--')
ax[-1].axhline(0.3,color='k',ls=':')
ax[-1].axhline(-0.3,color='k',ls=':')
graphics.reset_axes()

def update(k):
    graphics.plot_results(t_ind=k)
    graphics.plot_predictions(t_ind=k)

    ax[0].axhline(sp, color='r', ls='--', label='Reference')
    ax[1].axhline(0, color='r', ls='--')
    ax[2].axhline(0, color='r', ls='--')
    ax[3].axhline(0, color='r', ls='--')
    ax[4].axhline(0.3, color='k', ls=':')
    ax[4].axhline(-0.3, color='k', ls=':')

    # Labels
    ax[0].set_ylabel("Ball\nPosition [m]")
    ax[1].set_ylabel("Ball\nVelocity [m/s]")
    ax[2].set_ylabel(r"$\theta$ [rad]")
    ax[3].set_ylabel(r"$\omega$ [rad/s]")
    ax[4].set_ylabel(r"$\theta_c [rad]$")
    ax[4].set_xlabel("Time [s]")

    ax[0].set_title(f"Ball-on-Beam Nonlinear MPC (Step {k}), x0 = {x_initial:.2f}, sp = {sp:.2f}")

    for a in ax:
        a.grid(True, alpha=0.3)



ani = FuncAnimation(
    fig,
    update,
    frames=len(mpc.data['_time']), #type: ignore
    interval=100,
    blit=False
)

writer = PillowWriter(fps=8)
ani.save(
    "animations/ball_on_beam_mpc.gif",
    writer=writer
)

print(f"Saved animation to animations/ball_on_beam_mpc.gif")