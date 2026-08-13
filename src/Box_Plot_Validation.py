from utils import NODE_MPC_Sim,InitialSampler,SimParams,AnalyticalParams
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import gc
import csv
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser(description="Monte-Carlo Validation of NODE-MPC")
parser.add_argument(
    "--nCases",
    type = int,
    default = 30,
    help = "Number of cases to simulate per integrator and dt setting"
)

args= parser.parse_args()

nCases:int = args.nCases

experiments = [
    {"integrator_system":"euler","dt_system":0.01,"integrator_mpc":"euler","dt_mpc":0.01},
    {"integrator_system":"rk2","dt_system":0.01,"integrator_mpc":"rk2","dt_mpc":0.01},
    # {"integrator_system":"rk4","dt_system":0.01,"integrator_mpc":"rk4","dt_mpc":0.01}
]

sampler = InitialSampler()
aParams = AnalyticalParams()

initial_conditions = []

os.makedirs("validation_results", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

csv_path = (
    f"validation_results/"
    f"monte_carlo_results_{timestamp}.csv"
)

fieldnames = [
    "experiment_id",
    "case_id",

    "integrator_system",
    "dt_system",

    "integrator_mpc",
    "dt_mpc",
    "mpc_horizon",

    "x0",
    "setpoint",

    "total_solve_time",
    "mean_solve_time",

    "final_x",
    "final_v",
    "final_theta",
    "final_omega",

    "final_x_error",
    "final_position_error",

    "final_v_error",
    "final_theta_error",
    "final_omega_error",

    "final_state_error_norm",
]

with open(
    csv_path,
    "w",
    newline="",
    buffering=1
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

for i in range(nCases):

    x0, sp = sampler()

    initial_conditions.append(
        (x0, sp)
    )

nExperiments = len(experiments)

with open(
    csv_path,
    "a",
    newline="",
    buffering=1
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    for exp_idx, exp in enumerate(experiments):

        integrator_system = exp["integrator_system"]
        dt_system = exp["dt_system"]

        integrator_mpc = exp["integrator_mpc"]
        dt_mpc = exp["dt_mpc"]

        print("\n")
        print("=" * 70)
        print(
            f"EXPERIMENT {exp_idx + 1}/{nExperiments}: "
            f"System: {integrator_system.upper()}, dt={dt_system} "
            f"MPC: {integrator_mpc.upper()}, dt={dt_mpc}"
        )
        print("=" * 70)

        for case_idx in range(nCases):

            x0, sp = initial_conditions[case_idx]

            print(
                f"Case {case_idx + 1:03d}/{nCases}: "
                f"x0={x0:.4f}, sp={sp:.4f}",
                flush=True
            )

            Q = np.diag([
                275.0,
                250.0,
                100.0,
                150.0
            ])

            P = 5.0 * Q

            MPC_horizon = int(
                round(1.0 / dt_mpc)
            )

            params = SimParams(
                np.array([
                    x0,
                    0.0,
                    0.0,
                    0.0
                ]),

                np.array([
                    sp,
                    0.0,
                    0.0,
                    0.0
                ]),

                dt_system,
                dt_mpc,
                10.0,
                MPC_horizon,
                Q,
                P,
                1e-4,
                0.1,

                integrator_sys=integrator_system,
                integrator_mpc=integrator_mpc,

                STORE_FULL=False
            )

            obj = NODE_MPC_Sim(
                params,
                aParams,
                logging=False,
                verbose=False
            )

            obj.simulate()

            data = obj.data

            # ------------------------------------------------
            # Computation time
            # ------------------------------------------------

            solve_times = np.asarray(
                data["solve_time_history"]
            )

            total_time = np.sum(solve_times)
            average_time = np.mean(solve_times)

            # ------------------------------------------------
            # Final state
            # ------------------------------------------------

            final_state = np.asarray(
                data["beam_history"][-1]
            ).flatten()

            reference_state = np.array([
                sp,
                0.0,
                0.0,
                0.0
            ])

            error = final_state - reference_state

            final_position_error = abs(error[0])

            final_state_error_norm = np.linalg.norm(error)

            # ------------------------------------------------
            # Write result immediately
            # ------------------------------------------------

            writer.writerow({

                "experiment_id":
                    exp_idx + 1,

                "case_id":
                    case_idx + 1,

                "integrator_system":
                    integrator_system,

                "dt_system":
                    dt_system,

                "integrator_mpc":
                    integrator_mpc,

                "dt_mpc":
                    dt_mpc,

                "mpc_horizon":
                    MPC_horizon,

                "x0":
                    x0,

                "setpoint":
                    sp,

                "total_solve_time":
                    total_time,

                "mean_solve_time":
                    average_time,

                "final_x":
                    final_state[0],

                "final_v":
                    final_state[1],

                "final_theta":
                    final_state[2],

                "final_omega":
                    final_state[3],

                "final_x_error":
                    error[0],

                "final_position_error":
                    final_position_error,

                "final_v_error":
                    error[1],

                "final_theta_error":
                    error[2],

                "final_omega_error":
                    error[3],

                "final_state_error_norm":
                    final_state_error_norm,
            })

            print(
                f"    Final x = {final_state[0]:.6f}, "
                f"|error| = {final_position_error:.3e}, "
                f"solve = {total_time:.3f}s",
                flush=True
            )

            del obj
            gc.collect()

import pandas as pd

results = pd.read_csv(csv_path)

results["label"] = (
    results["integrator_system"].str.upper()
    + "\n"
    + results["dt_system"].astype(str)
    + "\n"
    + results["integrator_mpc"].str.upper()
    + "\n"
    + results["dt_mpc"].astype(str)
)

fig, ax = plt.subplots(figsize=(10, 6))

box_data = [
    results.loc[
        results["label"] == label,
        "total_solve_time"
    ].values
    for label in labels
]

ax.boxplot(
    box_data,
    tick_labels=labels
)

ax.set_ylabel("Total MPC solve time [s]")
ax.set_title(
    f"Total MPC Computation Time ({nCases} cases)"
)

ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "validation_results/total_solve_time_boxplot.png",
    dpi=300
)

plt.show()

fig, ax = plt.subplots(figsize=(10, 6))

box_data = [
    results.loc[
        results["label"] == label,
        "mean_solve_time"
    ].values
    for label in labels
]

ax.boxplot(
    box_data,
    tick_labels=labels
)

ax.set_ylabel("Mean MPC solve time per step [s]")
ax.set_title(
    f"Mean MPC Solve Time ({nCases} cases)"
)

ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "validation_results/mean_solve_time_boxplot.png",
    dpi=300
)

plt.show()

fig, ax = plt.subplots(figsize=(10, 6))

box_data = [
    results.loc[
        results["label"] == label,
        "final_position_error"
    ].values
    for label in labels
]

ax.boxplot(
    box_data,
    tick_labels=labels
)

ax.set_ylabel(r"$|x_f-x_{sp}|$ [m]")
ax.set_title(
    f"Final Position Error ({nCases} cases)"
)

ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "validation_results/final_position_error_boxplot.png",
    dpi=300
)

plt.show()

fig, ax = plt.subplots(figsize=(10, 6))

box_data = [
    results.loc[
        results["label"] == label,
        "final_state_error_norm"
    ].values
    for label in labels
]

ax.boxplot(
    box_data,
    tick_labels=labels
)

ax.set_ylabel(r"$\|X_f-X_{ref}\|_2$")
ax.set_title(
    f"Final State Error Norm ({nCases} cases)"
)

ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "validation_results/final_state_error_boxplot.png",
    dpi=300
)

plt.show()