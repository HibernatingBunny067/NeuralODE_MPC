import do_mpc
from casadi import vertcat
from CasAdi_Node import CasAdiMLP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os,time,gc

plt.rcParams['axes.grid'] = True
os.makedirs("animations",exist_ok=True)


#Parameters
x0 = 0.6
sp = 0.5
error0 = float(x0-sp)
cluster = None

if error0 > 0:
    cluster = 0
elif error0 < 0:
    cluster = 1
else:
    ValueError("Pick appropriate x0 and sp.")

print(F"Predicted Cluster: {cluster}")

node = CasAdiMLP(cluster=cluster) #type:ignore

# Initial Conditions
x0_ = np.array([x0,0.0,0.0,0.0]) 
reference_vector = np.array([sp,0.0,0.0,0.0])
T = 6 #secs
t_step = 0.1
N = int(T/t_step)
n_step_pred = 3
mpc_setup = {
    "n_horizon":n_step_pred,
    "t_step":t_step,
    "store_full_solution":True
}
running_weights = np.diag([200,50,50,5])
terminal_weights = running_weights*5

model = do_mpc.model.Model("continuous")

error = model.set_variable(
    var_type = "_x",
    var_name = "error"
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
    var_name= "theta_command",
)

errordot = node(
    vertcat(error,v,theta,omega),
    theta_command
)

model.set_rhs("error", errordot[0])
model.set_rhs("v", errordot[1])
model.set_rhs("theta", errordot[2])
model.set_rhs("omega", errordot[3])

model.setup()

mpc = do_mpc.controller.MPC(model)
mpc.set_param(**mpc_setup)

X = vertcat(error,v,theta,omega)

Q = running_weights
P = terminal_weights
reference_vector = vertcat(sp,0.0,0.0,0.0)
mterm = (X-reference_vector).T @ P @ (X-reference_vector)
lterm = (X-reference_vector).T @ Q @ (X-reference_vector) + theta_command*0.1*theta_command

mpc.set_objective(
    mterm = mterm,
    lterm=lterm
)

mpc.set_rterm(theta_command=1.0)

mpc.bounds["lower","_u","theta_command"] = -0.3
mpc.bounds["upper","_u","theta_command"] = 0.3

mpc.setup()

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = t_step)
simulator.setup()

simulator.x0 = x0_
mpc.x0 = x0_

mpc.set_initial_guess()

start = time.perf_counter()

print("|Starting Simulation|")
print("="*10)
for _ in range(N):
    u = mpc.make_step(x0_)
    x0_ = simulator.make_step(u)
end = time.perf_counter()

print(f"Finished the control problem in {end-start:.3f} seconds.")

graphics = do_mpc.graphics.Graphics(mpc.data)

fig,ax = plt.subplots(
    5,1,
    figsize=(12,10),
    sharex=True,
    constrained_layout = True
)

graphics.add_line("_x", "error", axis=ax[0])
graphics.add_line("_x", "v", axis=ax[1])
graphics.add_line("_x", "theta", axis=ax[2])
graphics.add_line("_x", "omega", axis=ax[3])
graphics.add_line("_u", "theta_command", axis=ax[4])

graphics.reset_axes()

def update(k):

    graphics.plot_results(t_ind=k)
    graphics.plot_predictions(t_ind=k)

    # References
    ax[0].axhline(sp, color='r', ls='--', label='Zero error')
    ax[1].axhline(0, color='r', ls='--')
    ax[2].axhline(0, color='r', ls='--')
    ax[3].axhline(0, color='r', ls='--')

    ax[4].axhline(0.3, color='k', ls=':')
    ax[4].axhline(-0.3, color='k', ls=':')

    ax[0].set_ylabel("Error [m]")
    ax[1].set_ylabel("Velocity")
    ax[2].set_ylabel(r"$\theta$")
    ax[3].set_ylabel(r"$\omega$")
    ax[4].set_ylabel(r"$\theta_c$")
    ax[4].set_xlabel("Time [s]")

    ax[0].set_title(
        f"Neural ODE MPC OpenLoop Test"
    )

    for a in ax:
        a.grid(True, alpha=0.3)

ani = FuncAnimation(
    fig,update,
    frames = len(mpc.data['_time']), #type:ignore
    interval = 100,
    blit=False
)

try:
    writer = FFMpegWriter(
        fps=8,
        metadata={"title": "Neural ODE MPC Animation"},
    )
    ani.save(f"animations/node_mpc_{cluster}_{x0}_{sp}.mp4", writer=writer)
    print("Animation saved at animations/node_mpc.mp4")
except Exception as exc:
    raise RuntimeError(
        "MP4 export failed. Install FFmpeg and ensure it is available on your PATH."
    ) from exc

gc.collect()