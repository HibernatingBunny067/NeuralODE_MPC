## <div align = "center">Data Generation Script for the NeuralODE

### State Space Equation for the Modelled System
$$
\begin{aligned}
\dot{x} &= v \\
\dot{v} &= -g\sin(\theta) + x\omega^2 - \mu \tanh(\frac{v}{\epsilon})(g\cos(\theta)+x\dot{\omega}+2v\omega) \\
\dot{\theta} &= \omega \\
\dot{\omega} &= \frac{-b\omega - k(\theta - \theta_{c}) - mgx\cos(\theta) - 2mv\omega x}{J + mx^2}
\end{aligned}
$$

#### Integrator Used: *RK45*
#### Total Time: *2.0 secs*
#### Sampling Time: *0.01 secs*
#### Total Unique Trajectories Sampled : *30K*

### Initial Conditions and Control Input Sampling Scheme
- Control input was taken constant for the entire time for a particular trajectory as the sampled scalar from the sampling distribution.

$$
\begin{aligned}
x_0 &\sim \text{Sobol}(0.03,0.97) \\ 
v_0 &\sim \mathcal{N}(0.0,0.02) \\ 
\theta_0 &\sim \mathcal{N}(0.0,0.02) \\ 
\omega_0 &\sim \mathcal{N}(0.0,0.01) \\ 
\theta_{c} &\sim \text{Sobol}(0.03,0.97)
\end{aligned}
$$