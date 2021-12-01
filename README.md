# Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control
This repo provides code for our paper "Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control".
Please check out our project website for more details: https://thaipduong.github.io/SE3HamDL/.

## Dependencies
Our code is tested with Ubuntu 18.04 and Python 3.7. It depends on the following Python packages: 

```torchdiffeq 0.1.1```

```gym 0.18.0```

```gym-pybullet-drones: https://github.com/utiasDSL/gym-pybullet-drones```

```torch 1.4.0``` and ```torch 1.9.0```

```numpy 1.20.1```

```scipy 1.5.3```

```matplotlib 3.3.4```

## Notes: For some reasons, the code does not work with ```torch 1.10.0```. Pending investiation.

## Demo with pendulum
Run ```python ./examples/pendulum/train_pend_SO3.py``` to train the model with data collected from the pendulum environment. It might take some time to train. A pretrained model is stored in ``` ./examples/pendulum/data/pendulum-so3ham_ode-rk4-5p.tar ```

Run ```python ./examples/pendulum/analyze_pend_SO3.py``` to plot the generalized mass inverse M^-1(q), the potential energy V(q), and the control coefficient g(q)
<p float="left">
<img src="figs/pendulum/M_x_all.png" height="180">
<img src="figs/pendulum/V_x.png" height="180">
<img src="figs/pendulum/g_x.png" height="180">
</p>

Run ```python ./examples/pendulum/rollout_pend_SO3.py``` to verify that our framework respect energy conservation and SE(3) constraints by construction, and plots phase portrait of a trajectory rolled out from our dynamics.
<p float="left">
<img src="figs/pendulum/hamiltonian.png" height="170">
<img src="figs/pendulum/SO3_constraints.png" height="170">
<img src="figs/pendulum/phase_portrait.png" height="170">
</p>

Run ```python ./examples/pendulum/control_pend_SO3.py``` to test the energy-based controller with the learned dynamics.
<p float="left">
<img src="figs/pendulum/pendulum_state_control.gif" width="400">
<img src="figs/pendulum/pendulum_animation_control.gif" width="300">
</p>


## Demo with quadrotor
Run ```python ./examples/quadrotor/train_quadrotor_SE3.py``` to train the model with data collected from the pybullet drone environment. It might take some time to train. A pretrained model is stored in ``` ./examples/quadrotor/data/quadrotor-se3ham-rk4-5p.tar ```
### Data collection
<p float="left">
<img src="figs/quadrotor/gif/data1.gif" width="200">
<img src="figs/quadrotor/gif/data2.gif" width="200">
<img src="figs/quadrotor/gif/data11.gif" width="200">
<img src="figs/quadrotor/gif/data15.gif" width="200">
<img src="figs/quadrotor/gif/data14.gif" width="200">
<img src="figs/quadrotor/gif/data13.gif" width="200">
<img src="figs/quadrotor/gif/data18.gif" width="200">
<img src="figs/quadrotor/gif/data21.gif" width="200">
<img src="figs/quadrotor/gif/data4.gif" width="200">
<img src="figs/quadrotor/gif/data9.gif" width="200">
<img src="figs/quadrotor/gif/data6.gif" width="200">
<img src="figs/quadrotor/gif/data19.gif" width="200">
</p>



Run ```python ./examples/quadrotor/analyze_quadrotor_SE3.py``` to plot the generalized mass inverse M^-1(q), the potential energy V(q), and the control coefficient g(q)
<p float="left">
<img src="figs/quadrotor/M1_x_all.png" height="180">
<img src="figs/quadrotor/M2_x_all.png" height="180">
<img src="figs/quadrotor/V_x.png" height="180">
<img src="figs/quadrotor/g_v_x.png" height="180">
<img src="figs/quadrotor/g_w_x.png" height="180">
</p>

Run ```python ./examples/quadrotor/rollout_quadrotor_SE3.py``` to verify that our framework respect energy conservation and SE(3) constraints by construction, and plots phase portrait of a trajectory rolled out from our dynamics.
<p float="left">
<img src="figs/quadrotor/hamiltonian.png" height="180">
<img src="figs/quadrotor/SE3_constraints.png" height="180">
</p>

Run ```python ./examples/quadrotor/control_quadrotor_SE3.py``` to test the energy-based controller with the learned dynamics.
<p float="left">
<img src="figs/quadrotor/gif/tracking_results.gif" height="250">
<img src="figs/quadrotor/gif/trajtracking.gif" height="250">
<img src="figs/quadrotor/gif/traj_results.gif" height="250">
</p>

## Citation
If you find our papers/code useful for your research, please cite our work as follows.

1. T. Duong, N. Atanasov. [Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control](https://thaipduong.github.io/SE3HamDL/). RSS, 2021

 ```bibtex
@inproceedings{duong21hamiltonian,
author = {Thai Duong AND Nikolay Atanasov},
title = {{Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control}},
booktitle = {Proceedings of Robotics: Science and Systems},
year = {2021},
address = {Virtual},
month = {July},
DOI = {10.15607/RSS.2021.XVII.086} 
}
```
