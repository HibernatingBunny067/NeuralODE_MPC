## Some example that were directly taken from the "do_mpc" documentation, used for learning the framework

- Primarily used for demonstrating, how simple MPC can be implemented on simple dynamical systems using the **do_mpc** Python toolbox.
- This folder is not part of the developed strategy.
- Some observations from the **do_mpc** working strategy,
    1. For the discritization, it defaults to orthogonal collocation (not the strategy used for training the NeuralODE in downstream workflow).
    2. Uses CasAdi for Non-Linear Programming
    3. Later, I implemented discritization and Numerical Integrators from scratch in pure CasAdi to resolve the fundamental training and inference policy conflict.