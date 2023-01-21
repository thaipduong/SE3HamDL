# Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control
This repo provides code for our paper "Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control".
Please check out our project website for more details: https://thaipduong.github.io/SE3HamDL/.

## Dependencies
Our code is tested with Ubuntu 18.04 and Python 3.7, Python 3.8. It depends on the following Python packages: 

```torchdiffeq 0.1.1, torchdiffeq 0.2.3```

```gym 0.18.0, gym 1.21.0```

```gym-pybullet-drones: https://github.com/utiasDSL/gym-pybullet-drones```

```torch 1.4.0, torch 1.9.0, torch 1.11.0```

```numpy 1.20.1```

```scipy 1.5.3```

```matplotlib 3.3.4```

```pyglet 1.5.27``` (pendulum rendering not working with pyglet >= 2.0.0)

***Notes: The NaN error during training with ```torch 1.10.0``` or newer has been fixed!!!!!!!!!.***



## Demo with 2D fully-actuated hexarotors
Run ```python ./examples/fadronesim_2d_bigdrone/train_fadrone_SE3_2D_bigdrone.py``` to train the model with data collected from the pybullet drone environment. It might take some time to train. A pretrained model is stored in ``` ./examples/fadronesim_2d_bigdrone/data/run6/fadronesim-se3ham-rk4-2p-final-5000.tar ```


Run ```python ./examples/quadrotor/analyze_quadrotor_SE3_2D.py``` to plot the generalized mass inverse M^-1(q), the potential energy V(q), and the control coefficient g(q)



## Citation
If you find our papers/code useful for your research, please cite our works as follows.

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

2. Z. Li, T. Duong, N. Atanasov, ***Robust and Safe Autonomous Navigation for Systems with Learned SE (3) Hamiltonian Dynamics***, IEEE Open Journal of Control Systems, 2022

 ```bibtex
@article{li2022robust,
  title={Robust and Safe Autonomous Navigation for Systems with Learned SE (3) Hamiltonian Dynamics},
  author={Li, Zhichao and Duong, Thai and Atanasov, Nikolay},
  journal={IEEE Open Journal of Control Systems},
  volume={1},
  pages={164--179},
  year={2022},
  publisher={IEEE}
}
}
```
