import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import do_mpc
from casadi import sin
import os

os.makedirs("animations",exist_ok=True)
plt.rcParams['axes.grid'] = True

# Model definition
model_type = "continuous"
model = do_mpc.model.Model(model_type)
x0 = np.array([0.5,0.0]).reshape(-1,1)
g,l = 9.81,1.0
t_step = 0.1
N = 100

# defining the states, theta theta_dot
theta = model.set_variable(
    var_type = "_x",
    var_name = "theta"
)

theta_dot = model.set_variable(
    var_type = "_x",
    var_name = "theta_dot"
)

u = model.set_variable(
    var_type = "_u",
    var_name = "u"
)

model.set_rhs("theta",theta_dot)
model.set_rhs("theta_dot",u+g*sin(theta)/l)
model.setup()

mpc = do_mpc.controller.MPC(model)
mpc_setup = {
    "n_horizon":30,
    "t_step":t_step,
    "store_full_solution":True
}

mpc.set_param(**mpc_setup)

lterm = 10*(theta)**2 + 2*theta_dot**2  + 0.1*u**2
mterm = 20*(theta)**2 + 5*theta_dot**2 

mpc.set_objective(
    mterm=mterm,
    lterm=lterm
)

mpc.set_rterm(u=0.5)

mpc.bounds['lower','_u',"u"] = -5
mpc.bounds['upper','_u','u'] = 5

mpc.setup()

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=t_step)
simulator.setup()

simulator.x0 = x0
mpc.x0 = x0

mpc.set_initial_guess()

for k in range(N):
    u = mpc.make_step(x0)
    x0 = simulator.make_step(u)

graphics = do_mpc.graphics.Graphics(mpc.data)

fig,axs = plt.subplots(3,1,figsize=(11,8),sharex=True)

graphics.add_line('_x','theta',axis=axs[0])
graphics.add_line('_x','theta_dot',axis=axs[1])
graphics.add_line('_u','u',axis=axs[2])

axs[0].axhline(0,color='r',ls='--')
axs[1].axhline(0,color='r',ls='--')

axs[2].axhline(5,color='k',ls=':')
axs[2].axhline(-5,color='k',ls=':')

graphics.reset_axes()

def update(k):
    graphics.plot_results(t_ind=k)
    graphics.plot_predictions(t_ind=k)

    axs[0].axhline(0,color='r',ls='--',label='Reference')
    axs[1].axhline(0,color='r',ls='--')
    axs[2].axhline(5,color='k',ls=':')
    axs[2].axhline(-5,color='k',ls=':')

    axs[0].set_ylabel(r"$\theta$")
    axs[1].set_ylabel(r"$\omega$")
    axs[2].set_ylabel(r"$u$")
    axs[2].set_xlabel("Time [s]")

    axs[0].set_title(f"MPC Step {k}")

    for ax in axs:
        ax.grid(True,alpha=0.3)


ani = FuncAnimation(
    fig,
    update,
    frames=len(mpc.data['_time']),
    interval=100,
    blit=False
)

gif_writer = PillowWriter(fps=5)
ani.save("animations/animation_INVERTED_PENDULUM.gif",writer = gif_writer)