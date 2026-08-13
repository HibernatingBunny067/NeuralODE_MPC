import os,time
import casadi
import do_mpc
from CasAdi_Node import CasAdiMLP
from numpy.typing import NDArray
from casadi import sin,cos, tanh, vertcat
import numpy as np
from scipy.stats import qmc
from dataclasses import dataclass,field
from datetime import datetime,UTC,timezone

class Integrators:
    def __init__(self,integrator:str="rk2",dt:float=0.01):
        self.choice = {
            "euler":self.euler_step,
            "rk2":self.rk2_step,
            "rk4":self.rk4_step
        }

        self.integrator = self.choice[integrator]

    def __call__(self,fnc,X,u,dt):
        return self.integrator(fnc,X,u,dt)


    def euler_step(self,fnc, state, u, dt):
        return state + dt * fnc(state, u)


    def rk2_step(self,fnc, state, u, dt):
        
        k1 = fnc(
            state,
            u
        )

        k2 = fnc(
            state + dt / 2 * k1,
            u
        )

        return state + dt * k2


    def rk4_step(self,fnc, state, u, dt):

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



class InitialSampler:
    def __init__(self,x0_range = np.array([0.15,0.75]),sp_range = np.array([0.15,0.75]),m=8):
        self.lb = np.array([
            x0_range[0],
            sp_range[0]
        ])
        self.ub = np.array([
            x0_range[-1],
            sp_range[-1]
        ])
        self.m = m

        self.__create_samples__()

    def __call__(self):
        idx = np.random.randint(0,2**self.m)

        while np.abs(self.samples[idx][0] - self.samples[idx][1]) <= 0.15:
            idx = np.random.randint(0,2**self.m)

        return (round(self.samples[idx][0],ndigits=2),round(self.samples[idx][1],ndigits=2))


    def __create_samples__(self):

        sampler = qmc.Sobol(
                    d=2,
                    scramble=True,
                    seed = 42
                )
        
        self.samples = qmc.scale(sampler.random_base2(m=self.m),self.lb,self.ub)

@dataclass
class SimParams:
    x0_:NDArray
    sp_:NDArray
    dt_system:float
    dt_mpc:float
    T_total:float
    horizon:int
    P_:NDArray
    Q_:NDArray
    u_term:float
    r_term:float
    u_bounds:NDArray = field(default_factory=lambda: np.array([-0.3,0.3]))
    x_bounds:NDArray = field( default_factory=lambda: np.array([0.05,0.85]))
    model_type:str = "discrete"
    integrator_sys:str = "rk4"
    integrator_mpc:str = "rk2"
    STORE_FULL:bool = False

@dataclass(frozen=True)
class AnalyticalParams:
    m_ball:float = 0.1
    mu_ball:float = 1e-3
    g:float = 9.81
    L_beam:float = 1.0
    J_beam:float = 0.5
    k_servo:float = 100.0
    b_beam:float = 10.0
    eps:float = 1e-4



class NODE_MPC_Sim:
    def __init__(
            self,
            params:SimParams,
            analyticalParams:AnalyticalParams,
            steps_recorded = 10,
            logging:bool = False,
            save_images:bool = False,
            verbose:bool = False) -> None:

        self.is_simulated = False
        self.params = params
        self.aParams = analyticalParams
        self.N = int(round(self.params.T_total/self.params.dt_system))
        self.steps_recorded = int(steps_recorded)

        self.MPC_PARAMS = {
            "n_horizon":params.horizon,
            "t_step":params.dt_mpc,
            "store_full_solution":params.STORE_FULL
        }
        
        self.NODE = CasAdiMLP(-1)
        self.plant = self.plant_rhs
        self.integrator_mpc = Integrators(params.integrator_mpc,dt = params.dt_mpc)
        self.integrator_sys = Integrators(params.integrator_sys,dt=params.dt_system)

        self.verbose = verbose
        self.logging = logging
        self.save_images = save_images

        self.ts = datetime.now(
                    timezone.utc
                ).strftime(
                    "%Y-%m-%d_%H-%M-%S"
                )
        if self.save_images:
            os.makedirs(f"animations/{self.ts}",exist_ok = True)

        if logging:

            os.makedirs(f"logs",exist_ok=True)
            self.log_file = f"logs/experiment_{self.ts}.txt"
            self.__init_txt_logger__()

        self.__init_neural_model__()
        self.__init_analytical_model__()
        self.__init_MPC__()
        self.__init_Simulator__()
        self.__init_Storage__()


    def __init_txt_logger__(self):
        with open(self.log_file,"w") as f:
            f.write("="*60)
            f.write("\n")
            f.write("NEURAL ODE MPC EXPERIMENT\n")
            f.write("="*60)
            f.write("\n\n")

            f.write(f"x0: {self.params.x0_[0]}\n")
            f.write(f"sp0: {self.params.sp_[0]}\n")
            f.write(f"v0: {self.params.x0_[1]}\n")
            f.write(f"theta0: {self.params.x0_[2]}\n")
            f.write(f"omega0: {self.params.x0_[3]}\n")

            f.write(f"dt: {self.params.dt_system}\n")
            f.write(f"total_time: {self.params.T_total}\n")
            f.write(f"MPC Horizon: {self.params.horizon} steps\n")
            f.write(
        f"MPC horizon time: "
        f"{self.params.horizon * self.params.dt_mpc:.3f} s\n"
    )   
            f.write(
        f"running weights:\n"
        f"{self.params.Q_}\n"
    )

            f.write(
        f"terminal weights:\n"
        f"{self.params.P_}\n"
    )
            f.write("\n")
            f.write(f"Integrator used in Plant: {self.params.integrator_sys}\n")
            f.write(f"Integrator used in NODE: {self.params.integrator_mpc}\n")
            f.write("=" * 60)
            f.write("\n")

    def __save_images__(self):
        pass

    def __report_metrics__(self):
        if self.is_simulated:
            time_vector = np.arange(self.N)*self.params.dt_system

            position = self.data['beam_history'][:,0]
            error = self.data['error_history']

            rmse = np.sqrt(np.mean(error**2))

            iae = np.sum(np.abs(error))*self.params.dt_system

            ise = np.sum(error**2)*self.params.dt_system

            if self.params.sp_[0] > self.params.x0_[0]:
                peak_position= np.max(position)
                overshoot = max(0.0,peak_position - self.params.sp_[0])

            else:
                peak_position= np.min(position)
                overshoot = max(0.0,-1*(peak_position - self.params.sp_[0]))
                
            control_effort = np.sum(self.data['control_history']**2)*self.params.dt_system

            control_variation = np.sum(np.abs(np.diff(self.data['control_history'])))

            mean_solve_time = np.mean(
                self.data['solve_time_history']
            )

            max_solve_time = np.max(
                self.data['solve_time_history']
            )


            settling_tolerance = 0.01

            settling_time = np.nan


            for i in range(self.N):

                if np.all(
                    np.abs(
                        error[i:]
                    ) <= settling_tolerance
                ):

                    settling_time = time_vector[i]

                    break
            
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

            with open(
                self.log_file,
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


    def simulate(self):
        if self.verbose:
            print("Starting SIMULATION....")

        measured_state = np.empty((4,1),dtype=np.float32)
        beam_state = self.params.x0_.copy()
        for i in range(self.N):

            step_start = time.perf_counter()

            measured_state = beam_state.copy()

            self.mpc.x0 = measured_state

            u = self.mpc.make_step(
                measured_state
            )

            solve_time = (time.perf_counter() - step_start)

            self.analytical_simulator.x0 = measured_state
            beam_state = self.analytical_simulator.make_step(u)

            self.data['beam_history'][i] = np.array(beam_state).flatten()
            self.data['control_history'][i] = np.array(u).flatten()[0]
            self.data['error_history'][i] = beam_state[0,0]-self.params.sp_[0]
            self.data['solve_time_history'][i] = solve_time

            if self.logging and i%self.steps_recorded == 0:
                with open(self.log_file,"a") as f:
                    f.write(
                f"step={i:04d}, "
                f"time={i*self.params.dt_system:.3f}, "
                f"x={beam_state[0,0]:.6f}, "
                f"error={self.data['error_history'][i]:.6f}, "
                f"u={self.data['control_history'][i]:.6f}, "
                f"solve_time={solve_time:.6f}\n"
            )

        self.is_simulated = True

        if self.logging:
            self.__report_metrics__()

        if self.save_images:
            self.__save_images__()


        #TODO 
        # 1. Add MPC prediction storage
        # 2. Metric display 

        

    def __init_Storage__(self):
        self.data ={
            "beam_history": np.zeros(
            (self.N,4),
            dtype=np.float32
            ),
            "control_history": np.zeros(self.N,dtype=np.float32),
            "error_history": np.zeros(self.N,dtype=np.float32),
            "solve_time_history":np.zeros(self.N,dtype=np.float32)
        }

    def __init_Simulator__(self):
        self.analytical_simulator = do_mpc.simulator.Simulator(self.analytical_model)

        self.analytical_simulator.set_param(t_step = self.params.dt_system)

        self.analytical_simulator.setup()

        self.analytical_simulator.x0 = self.params.x0_

        self.mpc.x0 = self.params.x0_

        self.mpc.set_initial_guess()



    def __init_MPC__(self):
        self.mpc = do_mpc.controller.MPC(self.neural_model)
        self.mpc.set_param(**self.MPC_PARAMS)

        #set objective function
        X_mpc = casadi.vertcat(self.x,self.v,self.theta,self.omega)
        reference = casadi.DM(self.params.sp_)

        mterm = (X_mpc-reference).T @ self.params.P_ @ (X_mpc-reference)
        lterm = (X_mpc-reference).T @ self.params.Q_ @ (X_mpc-reference) + self.params.u_term*self.theta_command**2

        self.mpc.set_objective(
            mterm=mterm,
            lterm=lterm
        )

        self.mpc.set_rterm(
            theta_command = self.params.r_term
        )

        self.mpc.bounds["upper","_u","theta_command"] = self.params.u_bounds[-1]
        self.mpc.bounds["lower","_u","theta_command"] = self.params.u_bounds[0]
        self.mpc.bounds["lower","_x","x"] = self.params.x_bounds[0]
        self.mpc.bounds["upper","_x","x"] = self.params.x_bounds[-1]

        self.mpc.setup()


    def __init_neural_model__(self):
        self.neural_model = do_mpc.model.Model(self.params.model_type)

        # define variables
        self.x = self.neural_model.set_variable(
            var_type = "_x",
            var_name= "x"
        )
        self.v = self.neural_model.set_variable(
            var_type= "_x",
            var_name="v"
        )
        self.theta = self.neural_model.set_variable(
            var_type="_x",
            var_name="theta"
        )
        self.omega = self.neural_model.set_variable(
            var_type="_x",
            var_name="omega"
        )
        self.theta_command = self.neural_model.set_variable(
            var_type = "_u",
            var_name="theta_command"
        )

        #set ODE
        X = casadi.vertcat(self.x,self.v,self.theta,self.omega)
        X_next = self.integrator_mpc(self.NODE,X,self.theta_command,dt=self.params.dt_mpc)

        self.neural_model.set_rhs("x",X_next[0])
        self.neural_model.set_rhs("v",X_next[1])
        self.neural_model.set_rhs("theta",X_next[2])
        self.neural_model.set_rhs("omega",X_next[3])

        self.neural_model.setup()
        

    def __init_analytical_model__(self):
        self.analytical_model = do_mpc.model.Model(self.params.model_type)

        #set variables
        x_p = self.analytical_model.set_variable(
            var_type = "_x",
            var_name= "x"
        )
        v_p = self.analytical_model.set_variable(
            var_type= "_x",
            var_name="v"
        )
        theta_p = self.analytical_model.set_variable(
            var_type="_x",
            var_name="theta"
        )
        omega_p = self.analytical_model.set_variable(
            var_type="_x",
            var_name="omega"
        )
        theta_command_p = self.analytical_model.set_variable(
            var_type = "_u",
            var_name="theta_command"
        )

        #set ODE

        X_plant = casadi.vertcat(x_p,v_p,theta_p,omega_p)
        X_next_analytical = self.integrator_sys(
            self.plant_rhs,
            X_plant,
            theta_command_p,
            dt=self.params.dt_system
        )

        self.analytical_model.set_rhs("x",X_next_analytical[0])
        self.analytical_model.set_rhs("v",X_next_analytical[1])
        self.analytical_model.set_rhs("theta",X_next_analytical[2])
        self.analytical_model.set_rhs("omega",X_next_analytical[3])

        self.analytical_model.setup()


    def plant_rhs(self,X:casadi.MX,u:casadi.SX):
        x = X[0]
        v = X[1]
        theta = X[2]
        omega = X[3]

        omega_dot_expr = (
        -self.aParams.b_beam * omega
        - self.aParams.k_servo * (theta - u)
        - self.aParams.m_ball * self.aParams.g * x * cos(theta)
        - 2 * self.aParams.m_ball * v * omega * x
    ) / (
        self.aParams.J_beam + self.aParams.m_ball * x**2
    )

        v_dot_expr = (
        -self.aParams.g * sin(theta)
        + x * omega**2
        - self.aParams.mu_ball * tanh(v / self.aParams.eps) * (
            self.aParams.g * cos(theta)
            + x * omega_dot_expr
            + 2 * v * omega
        )
    )

        return casadi.vertcat(v,v_dot_expr,omega,omega_dot_expr)


if __name__ == "__main__":

    sampler = InitialSampler()
    x0,sp = sampler()
    aParams = AnalyticalParams()
    Q = np.diag([
            275.0,
            175.0,
            100.0,
            150.0
        ])

    P = 5.0 * Q
    dt = 0.01

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

                0.01,
                0.01,
                10.0,
                100,
                Q,
                P,
                1e-4,
                0.1,
    
                integrator_sys="rk2",
                integrator_mpc= "rk2",

                STORE_FULL=False
            )

    obj = NODE_MPC_Sim(params,aParams,logging=True,save_images=True)

    obj.simulate()