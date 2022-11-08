import torch, os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.integrate
from scipy.spatial.transform import Rotation
from se3hamneuralode import L2_loss, from_pickle
from se3hamneuralode import MLP, PSD, SE3HamNODE
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from torchdiffeq import odeint_adjoint as odeint

gpu=0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = True

def plot_traj(traj, traj_hat, t_eval):
    fig = plt.figure(figsize=[10, 12])

    plt.subplot(3, 3, 1)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj_hat[:, 0], 'b--', label=r'$\hat R_11$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 2)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj_hat[:, 1], 'b--', label=r'$\hat R_12$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 3)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj[:, 2], 'b', label=r'$R_13$', linewidth=2)
    plt.plot(t_eval, traj_hat[:, 2], 'b--', label=r'$\hat R_13$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 4)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj_hat[:, 3], 'b--', label=r'$\hat R_21$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 5)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj_hat[:, 4], 'b--', label=r'$\hat R_22$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 6)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj[:, 5], 'b', label=r'$R_23$', linewidth=2)
    plt.plot(t_eval, traj_hat[:, 5], 'b--', label=r'$\hat R_23$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 7)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj[:, 6], 'b', label=r'$R_31$', linewidth=2)
    plt.plot(t_eval, traj_hat[:, 6], 'b--', label=r'$\hat R_31$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 8)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj[:, 7], 'b', label=r'$R_32$', linewidth=2)
    plt.plot(t_eval, traj_hat[:, 7], 'b--', label=r'$\hat R_32$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.subplot(3, 3, 9)
    plt.plot(t_eval, 1 * np.ones_like(t_eval), 'k--', linewidth=0.5)
    plt.plot(t_eval, 0 * np.ones_like(t_eval), 'k-', linewidth=0.5)
    plt.plot(t_eval, traj[:, 8], 'b', label=r'$R_33$', linewidth=2)
    plt.plot(t_eval, traj_hat[:, 8], 'b--', label=r'$\hat R_33$', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylim([-1.1, 1.1])
    plt.legend(fontsize=10)

    plt.tight_layout();
    plt.show()

def get_model():
    model =  SE3HamNODE(device=device, pretrain = False).to(device)
    path = 'data/quadrotor-se3ham-rk4-5p.tar'
    model.load_state_dict(torch.load(path, map_location=device))
    path = 'data/quadrotor-se3ham-rk4-5p-stats.pkl'
    stats = from_pickle(path)
    return model, stats

def get_init_state(rand = False):
    np_random, seed = gym.utils.seeding.np_random(None)
    rand_ = np_random.uniform(
        low=[-10, -10, 0, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, -0.5, -0.5, -0.5],
        high=[10, 10, 5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    x = rand_[0:3] if rand else np.array([0.0, 0.0, 0.0])  #
    x_dot = rand_[3:6] if rand else np.array([0.0, 0.0, 0.0])  #
    # Uniformly generate quaternion using http://planning.cs.uiuc.edu/node198.html
    u1, u2, u3 = rand_[6], rand_[7], rand_[8]
    quat = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                          np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)])
    omega =   rand_[9:12] if rand else np.array([0.0, 0.0, 0.0]) #
    r = Rotation.from_quat(quat)
    R = r.as_matrix()
    x_dot_bodyframe = np.matmul(np.transpose(R), x_dot)
    obs = np.hstack((x, R.flatten(), x_dot_bodyframe, omega))
    u0 = [0.0, 0.0, 0.0, 0.0]
    # State orders: x (3), R (9), linear vel (3), angular vel (3), control (4)
    y0_u = np.concatenate((obs, np.array(u0)))
    return y0_u

# Initial condition
import gym
from gym import wrappers

# time intervals
time_step = 100 ; n_eval = 100
t_span = [0,time_step*0.05]
t_eval = torch.linspace(t_span[0], t_span[1], n_eval)

# Get initial state
y0_u = get_init_state()
y0_u = torch.tensor(y0_u, requires_grad=True, device=device, dtype=torch.float64).view(1, 22)
y0_u = torch.cat((y0_u, y0_u), dim = 0)

# Roll out our dynamics model from the initial state
model, stats = get_model()
y = odeint(model, y0_u, t_eval, method='rk4')
y = y.detach().cpu().numpy()
y = y[:,0,:]
pose = torch.tensor(y[:,0:12], requires_grad=True, dtype=torch.float64).to(device)
x, R = torch.split(pose, [3, 9], dim=1)

# Get the output of the neural networks
g_q = model.g_net(pose)
V_q = model.V_net(pose)
M_q_inv1 = model.M_net1(x)
M_q_inv2 = model.M_net2(R)
V_q = V_q.detach().cpu().numpy()
M_q_inv1 = M_q_inv1.detach().cpu().numpy()
M_q_inv2 = M_q_inv2.detach().cpu().numpy()

# Calculate total energy from the learned dynamics
total_energy_learned = []
for i in range(len(M_q_inv1)):
    m1 = np.linalg.inv(M_q_inv1[i,:,:])
    m2 = np.linalg.inv(M_q_inv2[i,:,:])
    energy = 0.5*np.matmul(y[i,12:15].T, np.matmul(m1, y[i,12:15]))
    energy = energy + 0.5*np.matmul(y[i,15:18].T, np.matmul(m2, y[i,15:18]))
    energy = energy + V_q[i,0]
    total_energy_learned.append(energy)

# Plot the total energy
figsize = (12, 7.8)
fontsize = 24
fontsize_ticks = 32
line_width = 4
fig = plt.figure(figsize=figsize)
plt.plot(t_eval, total_energy_learned, 'r', linewidth=4, label='learned mass, potential, velocity')
plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
plt.ylim(total_energy_learned[0] - 0.5, total_energy_learned[0] + 0.5)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.legend(fontsize=fontsize)
plt.savefig('./png/hamiltonian.png', bbox_inches='tight', pad_inches=0.1)
plt.show()

# Check SE(3) constraints
det = []
RRT_I_dist = []
for i in range(len(y)):
    R_hat = y[i,3:12]
    R_hat = R_hat.reshape(3, 3)
    R_det = np.linalg.det(R_hat)
    det.append(np.abs(R_det - 1))
    R_RT = np.matmul(R_hat, R_hat.transpose())
    RRT_I = np.linalg.norm(R_RT - np.diag([1.0, 1.0, 1.0]))
    RRT_I_dist.append(RRT_I)

# Plot SE(3) constraints
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
ax.plot(t_eval, det, 'b', linewidth=line_width, label=r'$|det(R) - 1|$')
ax.plot(t_eval, RRT_I_dist, 'r', linewidth=line_width, label=r'$\Vert R R^\top - I\Vert$')
plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter(r'$%.0e$'))
plt.legend(fontsize=fontsize)
plt.savefig('./png/SE3_constraints.png', bbox_inches='tight')
plt.show()