from datetime import datetime, timezone
import argparse
import do_mpc
from casadi import vertcat, sin, cos, tanh
from CasAdi_Node import CasAdiMLP

import numpy as np
import matplotlib.pyplot as plt

import os
import time
import gc

parser = argparse.ArgumentParser(description="Neural ODE MPC Simulation")
parser.add_argument(
    "--intg",
    type=str,
    choices=['euler','rk2','rk4'],
    default='rk2',
    help='Numerical integrator used for both Neural ODE and analytical plant'
)
parser.add_argument(
    "--dt",
    type=float,
    default=0.01,
    help="Simulation and MPC timestep in seconds"
)
parser.add_argument(
    "--t_future",
    type = float,
    default = 1.0,
    help = "Seconds in the future the MPC looks from the internal model"
)

parser.add_argument(
    "--total_time",
    type = float,
    default = 10.0,
    help = "Total time of simulation"
)


args = parser.parse_args()

integrator = args.intg

ts = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%d_%H-%M-%S"
)


# ============================================================
# DIRECTORIES / PLOT SETTINGS
# ============================================================

plt.rcParams["axes.grid"] = True

os.makedirs(f"animations/{ts}", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ============================================================
# PHYSICAL PARAMETERS
# ============================================================

m_ball = 0.1
mu_ball = 1e-3
g = 9.81

L_beam = 1.0
J_beam = 0.5

k_servo = 100.0
b_beam = 10.0

eps = 1e-4


def euler_step(fnc, state, u, dt):
    return state + dt * fnc(state, u)


def rk2_step(fnc, state, u, dt):
    
    k1 = fnc(
        state,
        u
    )

    k2 = fnc(
        state + dt / 2 * k1,
        u
    )

    return state + dt * k2


def rk4_step(fnc, state, u, dt):

    k1 = fnc(
        state,
        u
    )

    k2 = fnc(
        state + dt / 2 * k1,
        u
    )

    k3 = fnc(
        state + dt / 2 * k2,
        u
    )

    k4 = fnc(
        state + dt * k3,
        u
    )

    return state + (
        dt / 6
        * (
            k1
            + 2 * k2
            + 2 * k3
            + k4
        )
    )

integrators = {
    "euler": euler_step,
    "rk2": rk2_step,
    "rk4": rk4_step,
}

integrator_step = integrators[integrator]

# ============================================================
# ANALYTICAL PLANT RHS
# ============================================================

def plant_rhs(X, u):

    x = X[0]
    v = X[1]
    theta = X[2]
    omega = X[3]

    omega_dot_expr = (
        -b_beam * omega
        - k_servo * (theta - u)
        - m_ball * g * x * cos(theta)
        - 2 * m_ball * v * omega * x
    ) / (
        J_beam + m_ball * x**2
    )

    v_dot_expr = (
        -g * sin(theta)
        + x * omega**2
        - mu_ball * tanh(v / eps) * (
            g * cos(theta)
            + x * omega_dot_expr
            + 2 * v * omega
        )
    )

    return vertcat(
        v,
        v_dot_expr,
        omega,
        omega_dot_expr
    )


# ============================================================
# INITIAL CONDITIONS
# ============================================================

x0 = 0.3
sp = 0.6

v0 = 0.0
theta0 = 0.0
omega0 = 0.0

error0 = x0 - sp

cluster = -1


beam_state = np.array(
    [x0, v0, theta0, omega0],
    dtype=np.float64
).reshape(-1, 1)

neural_state = beam_state.copy()


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

model_type = "discrete"

T_total = args.total_time
t_step = args.dt

N = int(T_total / t_step)

# MPC prediction horizon
n_step_predicted = int(round(args.t_future/t_step))

# Store one MPC prediction every N samples.
# Larger value -> less RAM.
prediction_save_interval = 10


# ============================================================
# MPC SETTINGS
# ============================================================

mpc_setup = {
    "n_horizon": n_step_predicted,
    "t_step": t_step,
}


# ============================================================
# COST FUNCTION WEIGHTS
# ============================================================

running_weights = np.diag([
    275.0,   # x
    175.0,   # v
    100.0,   # theta
    150.0    # omega
])

terminal_weights = running_weights * 5.0


# ============================================================
# CREATE MODELS
# ============================================================

neural_model = do_mpc.model.Model(model_type)
algebraic_model = do_mpc.model.Model(model_type)


# ============================================================
# 1. NEURAL ODE MODEL
# ============================================================

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


# Load Neural ODE
NODE = CasAdiMLP(cluster=cluster)


# ------------------------------------------------------------
# RK4 DISCRETIZATION OF NEURAL ODE
# ------------------------------------------------------------

X = vertcat(
    x,
    v,
    theta,
    omega
)

X_next = integrator_step(
    NODE,
    X,
    theta_command,
    dt=t_step
)


neural_model.set_rhs(
    "x",
    X_next[0]
)

neural_model.set_rhs(
    "v",
    X_next[1]
)

neural_model.set_rhs(
    "theta",
    X_next[2]
)

neural_model.set_rhs(
    "omega",
    X_next[3]
)

neural_model.setup()


# ============================================================
# 2. ANALYTICAL PLANT MODEL
# ============================================================

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


X_plant = vertcat(
    x_p,
    v_p,
    theta_p,
    omega_p
)


X_next_analytical = integrator_step(
    plant_rhs,
    X_plant,
    theta_command_p,
    dt=t_step
)


algebraic_model.set_rhs(
    "x",
    X_next_analytical[0]
)

algebraic_model.set_rhs(
    "v",
    X_next_analytical[1]
)

algebraic_model.set_rhs(
    "theta",
    X_next_analytical[2]
)

algebraic_model.set_rhs(
    "omega",
    X_next_analytical[3]
)

algebraic_model.setup()


# ============================================================
# MPC CONTROLLER
# ============================================================

mpc = do_mpc.controller.MPC(neural_model)

mpc.set_param(**mpc_setup)



# ============================================================
# MPC OBJECTIVE
# ============================================================

X_mpc = vertcat(
    x,
    v,
    theta,
    omega
)

reference_vector = vertcat(
    sp,
    0.0,
    0.0,
    0.0
)


Q = running_weights
P = terminal_weights


mterm = (
    (X_mpc - reference_vector).T
    @ P
    @ (X_mpc - reference_vector)
)


lterm = (
    (X_mpc - reference_vector).T
    @ Q
    @ (X_mpc - reference_vector)
    + 1e-4 * theta_command**2
)


mpc.set_objective(
    mterm=mterm,
    lterm=lterm
)


# Penalize changes in control input
mpc.set_rterm(
    theta_command=0.1
)


# ============================================================
# CONSTRAINTS
# ============================================================

mpc.bounds[
    "lower",
    "_u",
    "theta_command"
] = -0.3

mpc.bounds[
    "upper",
    "_u",
    "theta_command"
] = 0.3


mpc.bounds[
    "lower",
    "_x",
    "x"
] = 0.05

mpc.bounds[
    "upper",
    "_x",
    "x"
] = L_beam - 0.05

mpc.bounds[
    "upper",
    "_x",
    "v"
] = 0.5


mpc.setup()


# ============================================================
# ANALYTICAL PLANT SIMULATOR
# ============================================================

algebraic_simulator = do_mpc.simulator.Simulator(
    algebraic_model
)

algebraic_simulator.set_param(
    t_step=t_step
)

algebraic_simulator.setup()

algebraic_simulator.x0 = beam_state


# ============================================================
# MPC INITIALIZATION
# ============================================================

mpc.x0 = neural_state
mpc.set_initial_guess()


# ============================================================
# HISTORY ARRAYS
# ============================================================

beam_history = np.zeros(
    (N, 4)
)

control_history = np.zeros(
    N
)

error_history = np.zeros(
    N
)

solve_time_history = np.zeros(
    N
)


# ============================================================
# SPARSE MPC PREDICTION STORAGE
# ============================================================

prediction_snapshots = []
prediction_times = []


# ============================================================
# LOGGING
# ============================================================



log_file = f"logs/experiment_{ts}.txt"


with open(log_file, "w") as f:

    f.write("=" * 60)
    f.write("\n")
    f.write("NEURAL ODE MPC EXPERIMENT\n")
    f.write("=" * 60)
    f.write("\n\n")

    f.write(f"x0: {x0}\n")
    f.write(f"setpoint: {sp}\n")

    f.write(f"v0: {v0}\n")
    f.write(f"theta0: {theta0}\n")
    f.write(f"omega0: {omega0}\n")


    f.write(f"dt: {t_step}\n")
    f.write(f"total_time: {T_total}\n")

    f.write(f"MPC horizon: {n_step_predicted}\n")
    f.write(
        f"MPC horizon time: "
        f"{n_step_predicted * t_step:.3f} s\n"
    )

    f.write(
        f"running weights:\n"
        f"{running_weights}\n"
    )

    f.write(
        f"terminal weights:\n"
        f"{terminal_weights}\n"
    )

    f.write("\n")
    f.write(f"Integrator used: {args.intg}\n")
    f.write("=" * 60)
    f.write("\n")


# ============================================================
# CLOSED LOOP SIMULATION
# ============================================================

print("\nStarting closed-loop simulation...\n")

start = time.perf_counter()

horizon_changed = False
for i in range(N):

    step_start = time.perf_counter()


    # --------------------------------------------------------
    # MEASUREMENT
    # --------------------------------------------------------

    measured_state = beam_state.copy()



    # Controller model starts from measured plant state
    neural_state = measured_state.copy()

    mpc.x0 = neural_state


    # --------------------------------------------------------
    # MPC OPTIMIZATION
    # --------------------------------------------------------

    u = mpc.make_step(
        neural_state
    )


    # --------------------------------------------------------
    # COMPUTATIONAL TIME
    # --------------------------------------------------------

    solve_time = (
        time.perf_counter()
        - step_start
    )


    # --------------------------------------------------------
    # ADVANCE ACTUAL PLANT
    # --------------------------------------------------------

    algebraic_simulator.x0 = measured_state

    beam_state = algebraic_simulator.make_step(
        u
    )


    # --------------------------------------------------------
    # STORE CLOSED-LOOP DATA
    # --------------------------------------------------------

    beam_history[i] = (
        np.array(
            beam_state
        ).flatten()
    )

    control_history[i] = float(
        np.array(u).flatten()[0]
    )

    error_history[i] = (
        beam_state[0, 0] - sp
    )

    solve_time_history[i] = solve_time


    # --------------------------------------------------------
    # SAVE SPARSE MPC PREDICTIONS
    # --------------------------------------------------------

    if i % prediction_save_interval == 0:

        try:

            predicted_states = np.array(
                mpc.opt_x_num["_x"]
            )

            prediction_snapshots.append(
                predicted_states.copy()
            )

            prediction_times.append(
                i * t_step
            )

        except Exception as exc:

            if i == 0:

                print(
                    "Warning: Could not extract "
                    "MPC prediction from opt_x_num:"
                )

                print(exc)


    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    if i % 10 == 0:

        with open(
            log_file,
            "a"
        ) as f:

            f.write(
                f"step={i:04d}, "
                f"time={i*t_step:.3f}, "
                f"x={beam_state[0,0]:.6f}, "
                f"error={error_history[i]:.6f}, "
                f"u={control_history[i]:.6f}, "
                f"solve_time={solve_time:.6f}\n"
            )


end = time.perf_counter()


print(
    f"Simulation completed in "
    f"{end-start:.3f} seconds."
)


# ============================================================
# TIME VECTOR
# ============================================================

time_vector = (
    np.arange(N)
    * t_step
)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

position = beam_history[:, 0]

error = error_history


# RMSE
rmse = np.sqrt(
    np.mean(error**2)
)


# Integral Absolute Error
iae = np.sum(
    np.abs(error)
) * t_step


# Integral Squared Error
ise = np.sum(
    error**2
) * t_step


# Peak overshoot relative to reference
if sp > x0:

    peak_position = np.max(position)

    overshoot = max(
        0.0,
        peak_position - sp
    )

else:

    peak_position = np.min(position)

    overshoot = max(
        0.0,
        sp - peak_position
    )


# Control effort
control_effort = np.sum(
    control_history**2
) * t_step


# Total variation in control
control_variation = np.sum(
    np.abs(
        np.diff(control_history)
    )
)


# Computational statistics
mean_solve_time = np.mean(
    solve_time_history
)

max_solve_time = np.max(
    solve_time_history
)


# ============================================================
# SETTLING TIME
# ============================================================

settling_tolerance = 0.01

settling_time = np.nan


for i in range(N):

    if np.all(
        np.abs(
            error[i:]
        ) <= settling_tolerance
    ):

        settling_time = time_vector[i]

        break


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CLOSED-LOOP PERFORMANCE")
print("=" * 60)

print(
    f"RMSE              : {rmse:.6f} m"
)

print(
    f"IAE               : {iae:.6f} m.s"
)

print(
    f"ISE               : {ise:.6f} m².s"
)

print(
    f"Peak position      : {peak_position:.6f} m"
)

print(
    f"Peak overshoot     : {overshoot:.6f} m"
)

if np.isnan(settling_time):

    print(
        "Settling time      : Not settled"
    )

else:

    print(
        f"Settling time      : "
        f"{settling_time:.3f} s"
    )

print(
    f"Control effort     : "
    f"{control_effort:.6f}"
)

print(
    f"Control variation  : "
    f"{control_variation:.6f}"
)

print(
    f"Mean MPC solve     : "
    f"{mean_solve_time*1000:.3f} ms"
)

print(
    f"Max MPC solve      : "
    f"{max_solve_time*1000:.3f} ms"
)

print("=" * 60)


# ============================================================
# SAVE METRICS TO LOG
# ============================================================

with open(
    log_file,
    "a"
) as f:

    f.write("\n")
    f.write("=" * 60)
    f.write("\n")
    f.write("PERFORMANCE METRICS\n")
    f.write("=" * 60)
    f.write("\n")

    f.write(f"RMSE: {rmse}\n")
    f.write(f"IAE: {iae}\n")
    f.write(f"ISE: {ise}\n")
    f.write(f"Peak position: {peak_position}\n")
    f.write(f"Peak overshoot: {overshoot}\n")
    f.write(f"Settling time: {settling_time}\n")
    f.write(f"Control effort: {control_effort}\n")
    f.write(f"Control variation: {control_variation}\n")
    f.write(
        f"Mean solve time: "
        f"{mean_solve_time}\n"
    )
    f.write(
        f"Max solve time: "
        f"{max_solve_time}\n"
    )


# ============================================================
# FIGURE 1
# CLOSED-LOOP RESPONSE
# ============================================================

fig1, ax = plt.subplots(
    5,
    1,
    figsize=(13, 12),
    sharex=True
)

run_name = (
    f"{args.intg}"
    f"_dt{t_step:g}"
    f"_H{args.t_future:g}"
    f"_x0{x0:g}"
    f"_sp{sp:g}"
)

# ------------------------------------------------------------
# Position
# ------------------------------------------------------------

ax[0].plot(
    time_vector,
    position,
    label="Analytical plant"
)

ax[0].axhline(
    sp,
    color="r",
    linestyle=":",
    label="Reference"
)

ax[0].axhline(
    sp + settling_tolerance,
    color="k",
    linestyle="--",
    alpha=0.4
)

ax[0].axhline(
    sp - settling_tolerance,
    color="k",
    linestyle="--",
    alpha=0.4
)

ax[0].fill_between(
    time_vector,
    sp - settling_tolerance,
    sp + settling_tolerance,
    alpha=0.08
)

ax[0].set_ylabel("x [m]")
ax[0].set_title("Closed-Loop Position")
ax[0].legend()


# ------------------------------------------------------------
# Tracking Error
# ------------------------------------------------------------

ax[1].plot(
    time_vector,
    error,
    label="Tracking error"
)

ax[1].axhline(
    0.0,
    color="r",
    linestyle=":"
)

ax[1].axhline(
    settling_tolerance,
    color="k",
    linestyle="--",
    alpha=0.4
)

ax[1].axhline(
    -settling_tolerance,
    color="k",
    linestyle="--",
    alpha=0.4
)

ax[1].fill_between(
    time_vector,
    -settling_tolerance,
    settling_tolerance,
    alpha=0.08
)

ax[1].set_ylabel("e [m]")
ax[1].set_title("Tracking Error")
ax[1].legend()


# ------------------------------------------------------------
# Velocity
# ------------------------------------------------------------

ax[2].plot(
    time_vector,
    beam_history[:, 1],
    label="Ball velocity"
)

ax[2].axhline(
    0.0,
    color="r",
    linestyle=":"
)

ax[2].set_ylabel("v [m/s]")
ax[2].set_title("Ball Velocity")
ax[2].legend()


# ------------------------------------------------------------
# Beam angle
# ------------------------------------------------------------

ax[3].plot(
    time_vector,
    beam_history[:, 2],
    label=r"$\theta$"
)

ax[3].axhline(
    0.0,
    color="r",
    linestyle=":"
)

ax[3].set_ylabel(r"$\theta$ [rad]")
ax[3].set_title("Beam Angle")
ax[3].legend()


# ------------------------------------------------------------
# Angular velocity
# ------------------------------------------------------------

ax[4].plot(
    time_vector,
    beam_history[:, 3],
    label=r"$\omega$"
)

ax[4].axhline(
    0.0,
    color="r",
    linestyle=":"
)

ax[4].set_ylabel(
    r"$\omega$ [rad/s]"
)

ax[4].set_xlabel("Time [s]")
ax[4].set_title("Beam Angular Velocity")
ax[4].legend()


fig1.suptitle(
    f"Neural ODE MPC — Closed-Loop Response - {integrator}",
    fontsize=15
)

plt.tight_layout()

fig1.savefig(
    f"animations/{ts}/closed_loop_{run_name}.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# FIGURE 2
# CONTROL + COMPUTATIONAL PERFORMANCE
# ============================================================

fig2, ax2 = plt.subplots(
    3,
    1,
    figsize=(13, 9),
    sharex=True
)


# ------------------------------------------------------------
# Control
# ------------------------------------------------------------

ax2[0].plot(
    time_vector,
    control_history,
    label=r"$\theta_c$"
)

ax2[0].axhline(
    0.3,
    color="r",
    linestyle="--",
    alpha=0.5
)

ax2[0].axhline(
    -0.3,
    color="r",
    linestyle="--",
    alpha=0.5
)

ax2[0].axhline(
    0.0,
    color="k",
    linestyle=":"
)

ax2[0].set_ylabel(
    r"$\theta_c$ [rad]"
)

ax2[0].set_title("Control Input")
ax2[0].legend()


# ------------------------------------------------------------
# MPC solve time
# ------------------------------------------------------------

ax2[1].plot(
    time_vector,
    solve_time_history * 1000
)

ax2[1].axhline(
    mean_solve_time * 1000,
    color="r",
    linestyle=":",
    label="Mean"
)

ax2[1].set_ylabel(
    "Solve time [ms]"
)

ax2[1].set_title(
    "MPC Computational Time"
)

ax2[1].legend()


# ------------------------------------------------------------
# Absolute error
# ------------------------------------------------------------

ax2[2].plot(
    time_vector,
    np.abs(error)
)

ax2[2].axhline(
    settling_tolerance,
    color="r",
    linestyle=":",
    label="±1 cm threshold"
)

ax2[2].set_ylabel(
    "|e| [m]"
)

ax2[2].set_xlabel(
    "Time [s]"
)

ax2[2].set_title(
    "Absolute Tracking Error"
)

ax2[2].legend()


fig2.suptitle(
    "Neural ODE MPC — Control and Computational Performance",
    fontsize=15
)

plt.tight_layout()

fig2.savefig(
    # f"animations/{ts}"
    # f"control_performance_{n_step_predicted}_"
    # f"{x0}_{sp}.png",
    f"animations/{ts}/control_performance_{run_name}.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# FIGURE 3
# MPC PREDICTION SNAPSHOTS
# ============================================================

if len(prediction_snapshots) > 0:

    fig3, ax3 = plt.subplots(
        figsize=(13, 6)
    )

    ax3.plot(
        time_vector,
        position,
        linewidth=2,
        label="Actual plant"
    )

    ax3.axhline(
        sp,
        color="r",
        linestyle=":",
        linewidth=2,
        label="Reference"
    )


    # --------------------------------------------------------
    # Plot sparse prediction trajectories
    # --------------------------------------------------------

    for prediction, t0 in zip(
        prediction_snapshots,
        prediction_times
    ):

        try:

            prediction = np.asarray(
                prediction
            )

            # Depending on do-mpc's structure,
            # remove unnecessary dimensions.

            prediction = np.squeeze(
                prediction
            )

            # If the complete state prediction
            # is available, x is the first state.

            if prediction.ndim == 2:

                predicted_x = prediction[:, 0]

            else:

                predicted_x = prediction


            prediction_time = (
                t0
                + np.arange(
                    len(predicted_x)
                ) * t_step
            )


            ax3.plot(
                prediction_time,
                predicted_x,
                alpha=0.20
            )

        except Exception:

            pass


    ax3.set_xlabel(
        "Time [s]"
    )

    ax3.set_ylabel(
        "Ball position [m]"
    )

    ax3.set_title(
        "Neural ODE MPC — Receding-Horizon Predictions"
    )

    ax3.legend()

    fig3.tight_layout()

    fig3.savefig(
        # f"animations/{ts}"
        # f"mpc_predictions_{n_step_predicted}_"
        # f"{x0}_{sp}.png",
        f"animations/{ts}/mpc_predictions_{run_name}.png",
        dpi=300,
        bbox_inches="tight"
    )


# ============================================================
# FINAL DISPLAY
# ============================================================

plt.show()


# ============================================================
# CLEANUP
# ============================================================

gc.collect()