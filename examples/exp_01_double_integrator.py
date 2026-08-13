import do_mpc
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
import warnings
from matplotlib.animation import FuncAnimation,PillowWriter
warnings.filterwarnings("ignore")
import os

os.makedirs("animations",exist_ok=True)
# Model definition
model_type = "continuous"
model = do_mpc.model.Model(model_type)
x0 = np.array([2,-2]).reshape(-1,1)
t_step = 0.1

## defining the variables

x1 = model.set_variable(
    var_type = "_x",
    var_name = "x1"
)

x2 = model.set_variable(
    var_type = "_x",
    var_name = "x2"
)

u = model.set_variable(
    var_type = "_u",
    var_name = "u"
)

# defining the equations

model.set_rhs("x1",x2)
model.set_rhs("x2",u)

model.setup()


# defining the controller
mpc = do_mpc.controller.MPC(model)

mpc_setup = {
    "n_horizon":10,
    "t_step":t_step,
    "store_full_solution":True
}

mpc.set_param(**mpc_setup)

# defining the loss function for the MPC
mterm = 50*(x1-1)**2 + 5*x2**2 
lterm = 20*(x1-1)**2 + x2**2 + 0.1*u**2

mpc.set_objective(
    mterm=mterm,
    lterm=lterm
)

mpc.set_rterm(u=1.0)

mpc.bounds['lower','_u','u'] = -1.5
mpc.bounds['upper','_u','u'] = 1.5

mpc.setup()

## Setting up the simulator
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = 0.1)
simulator.setup()

simulator.x0 = x0
mpc.x0 = x0

mpc.set_initial_guess()



for k in range(50):
    u = mpc.make_step(x0)
    x0 = simulator.make_step(u)
    
graphics = do_mpc.graphics.Graphics(mpc.data)

fig, axs = plt.subplots(
    3,
    1,
    figsize=(11,8),
    sharex=True,
    constrained_layout=True
)

graphics.add_line('_x','x1',axis=axs[0])
graphics.add_line('_x','x2',axis=axs[1])
graphics.add_line('_u','u',axis=axs[2])
axs[0].axhline(1,color='r',ls='--')
axs[1].axhline(0,color='r',ls='--')

axs[2].axhline(2,color='k',ls=':')
axs[2].axhline(-2,color='k',ls=':')

graphics.reset_axes()

def update(k):


    graphics.plot_results(t_ind=k)
    graphics.plot_predictions(t_ind=k)

    axs[0].axhline(1,color='r',ls='--',label='Reference')
    axs[1].axhline(0,color='r',ls='--')
    axs[2].axhline(2,color='k',ls=':')
    axs[2].axhline(-2,color='k',ls=':')

    axs[0].set_ylabel(r"$x_1$")
    axs[1].set_ylabel(r"$x_2$")
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
ani.save("animations/animation_DOUBLE_INTEGRATOR.gif",writer = gif_writer)