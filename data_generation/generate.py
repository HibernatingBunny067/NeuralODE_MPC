import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import qmc

# ===========================================================
# Parameters
# ===========================================================

N_TRAJ = 30000

T = 2.0 
dt = 0.01

m_ball = 0.1
mu_ball = 1e-3
g = 9.81
J_beam = 0.5
k_servo = 100.0
b_beam = 10.0
eps = 1e-4

rng = np.random.default_rng(42)

all_data = []

# ===========================================================
# Sobol Sampling (x0, u)
# ===========================================================

sampler = qmc.Sobol(
    d=2,
    scramble=True,
    seed=42
)

# Generate a power-of-two number of samples
m = int(np.ceil(np.log2(N_TRAJ)))
samples = sampler.random_base2(m=m)

# Keep only what we need
samples = samples[:N_TRAJ]

lb = np.array([
    0.03,    # x0
   -0.35     # control
])

ub = np.array([
    0.97,
    0.35
])

samples = qmc.scale(samples, lb, ub)

# ===========================================================
# Continuous dynamics
# ===========================================================

def dynamics(t, X, control):

    x, v, theta, omega = X

    u = control(t)

    omega_dot = (
        -b_beam * omega
        - k_servo * (theta - u)
        - m_ball * g * x * np.cos(theta)
        - 2.0 * m_ball * v * omega * x
    ) / (J_beam + m_ball * x**2)

    v_dot = (
        -g * np.sin(theta)
        + x * omega**2
        - mu_ball * np.tanh(v / eps)
        * (
            g * np.cos(theta)
            + x * omega_dot
            + 2.0 * v * omega
        )
    )

    return np.array([
        v,
        v_dot,
        omega,
        omega_dot
    ])

# ===========================================================
# Stop integration once ball leaves beam
# ===========================================================

def beam_exit_event(t, X, control):
    x = X[0]
    return min(x, 1.0 - x)

beam_exit_event.terminal = True
beam_exit_event.direction = -1

# ===========================================================
# Dataset generation
# ===========================================================

print("Generating trajectories...\n")

start = time.perf_counter()

traj = 0

while traj < N_TRAJ:

    # -----------------------------
    # Sobol sampled variables
    # -----------------------------
    x0, u_const = samples[traj]

    # -----------------------------
    # Gaussian variables
    # -----------------------------
    v0 = rng.normal(0.0, 0.02)
    theta0 = rng.normal(0.0, 0.02)
    omega0 = rng.normal(0.0, 0.01)

    # if x0 > 1.0 or x0 < 0.0:
    #     continue

    X0 = np.array([
        x0,
        v0,
        theta0,
        omega0
    ])

    control = lambda t, u=u_const: u

    t_eval = np.arange(
        0.0,
        T + dt,
        dt
    )

    sol = solve_ivp(
        lambda t, x: dynamics(t, x, control),
        (0.0, T),
        X0,
        t_eval=t_eval,
        events=lambda t, x: beam_exit_event(t, x, control),
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        continue

    # Reject trajectories that leave almost immediately
    if len(sol.t) < 125:
        continue

    u = np.full_like(sol.t, u_const)

    df = pd.DataFrame({

        "traj_id": traj,

        "time": sol.t,

        "x": sol.y[0],
        "v": sol.y[1],
        "theta": sol.y[2],
        "omega": sol.y[3],

        "theta_command": u,

        "x0": x0,
        "v0": v0,
        "theta0": theta0,
        "omega0": omega0

    })

    all_data.append(df)

    traj += 1

    if traj % 1000 == 0:
        print(f"{traj}/{N_TRAJ}")

end = time.perf_counter()

# ===========================================================
# Save dataset
# ===========================================================

dataset = pd.concat(
    all_data,
    ignore_index=True
)

dataset.to_csv(
    "open_loop_dataset_exit.csv",
    index=False
)

print("\nFinished!")
print(f"Generated {traj} trajectories.")
print(f"Total samples : {len(dataset)}")
print(f"Elapsed time  : {end-start:.2f} s")

print("\nDataset preview:\n")
print(dataset.head())