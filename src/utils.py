from casadi import sin,cos, tanh, vertcat

class Integrators:
    def __init__(self,integrator:str="rk2",dt:float=0.01):
        self.choice = {
            "euler":self.euler_step,
            "rk2":self.rk2_step,
            "rk4":self.rk4_step
        }

        self.integrator = self.choice[integrator]

    def __call__(self,fnc,X,u,dt):
        return self.choice[self.integrator](fnc,X,u,dt)



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

def plant_rhs(X,u,
              b_beam=10.0,
              k_servo=100.0,
              J_beam=0.5,
              m_ball=0.1,
              mu_ball=1e-3,
              eps = 1e-4,
              g=9.81):
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

