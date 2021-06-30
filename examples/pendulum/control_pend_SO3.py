import torch, os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import gym
from scipy.spatial.transform import Rotation
from se3hamneuralode import from_pickle, SO3HamNODE
solve_ivp = scipy.integrate.solve_ivp
from gym import wrappers
import envs

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = True

gpu=0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

def plot_traj(traj, t_eval):
    ''' Plotting trajectory'''
    # Figure and font size
    figsize = (12, 7.8)
    fontsize = 24
    fontsize_ticks = 32
    linewidth = 4

    traj = np.array(traj)

    fig = plt.figure(figsize= figsize)
    plt.plot(t_eval, traj[:, 0], label=r'$\varphi$', linewidth=linewidth)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlabel('$t(s)$', fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('./png/control_theta.png', bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize= figsize)
    plt.plot(t_eval, traj[:, 1], label=r'$\dot{\varphi}$', linewidth=linewidth)
    plt.xlabel('$t(s)$', fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('./png/control_thetadot.png', bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=figsize)
    plt.plot(t_eval, traj[:, -1], label=r'$u$', linewidth=linewidth)
    plt.xlabel('$t(s)$', fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('./png/control_input.png', bbox_inches='tight')
    plt.show()

def get_model():
    model = SO3HamNODE(device=device, u_dim=1).to(device)
    path = './data/pendulum-so3ham_ode-rk4-5p.tar'
    model.load_state_dict(torch.load(path, map_location=device))
    path = './data/pendulum-so3ham_ode-rk4-5p-stats.pkl'
    stats = from_pickle(path)
    return model, stats

# Load trained model
model, stats = get_model()

# Time info
time_step = 300 ; n_eval = 300
t_span = [0,time_step*0.05]
t_eval = torch.linspace(t_span[0], t_span[1], n_eval)
# Init angle and control
init_angle = 0.0
u0 = 0.0

# Create and reset the pendulum environment to the initialized values.
env = gym.make('MyPendulum-v1')
# Record video
env = gym.wrappers.Monitor(env, './videos/' + 'pendulum' + '/', force=True) # , video_callable=lambda x: True, force=True
env.reset(ori_rep='rotmat')
env.env.state = np.array([init_angle, u0], dtype=np.float32)
obs = env.env._get_obs()
# Get state as input for the neural networks
y = np.concatenate((obs, np.array([u0])))
y = torch.tensor(y, requires_grad=True, device=device, dtype=torch.float32).view(1, 13)

# Desired state
rd = Rotation.from_euler('xyz', [0.0, 0.0, 3.14])
rd_matrix = rd.as_matrix()
R_d = torch.unsqueeze(torch.tensor(rd_matrix, device=device, dtype=torch.float32), dim = 0)
R_d = R_d.view(-1,9)

# Start controller
# Controller gain
k_d = 0.4
K_p = 2.0

state_traj = []
s = env.get_state()
s = np.concatenate((s, np.array([u0])))
state_traj.append(s)
frames = []
for i in range(len(t_eval)-1):
    frames.append(env.render(mode='rgb_array'))
    # Get states q = R and q_dot
    R, q_dot, _ = torch.split(y, [9, 3, 1], dim=1)
    q_dot = torch.unsqueeze(q_dot, dim=2)
    # Get M^-1, V, and g values from the neural networks.
    M_inv = model.M_net(R)
    V_q = torch.unsqueeze(model.V_net(R), dim=2)
    g_q = torch.unsqueeze(model.g_net(R), dim = 2)
    dV_q = torch.autograd.grad(V_q, R)[0]
    # Calculate momenta p
    p = torch.squeeze(torch.matmul(torch.inverse(M_inv), q_dot), dim=2)
    p = torch.unsqueeze(p, dim=2)

    # Calculate control input u
    dH_a = torch.cross(R[:, 0:3], -dV_q[:, 0:3] - 0.5*K_p*R_d[:, 0:3]) \
           + torch.cross(R[:, 3:6], -dV_q[:, 3:6] - 0.5*K_p*R_d[:, 3:6]) \
           + torch.cross(R[:, 6:9], -dV_q[:, 6:9] - 0.5*K_p*R_d[:, 6:9])
    dH_a = torch.unsqueeze(dH_a, dim=2)
    gTg = torch.matmul(torch.transpose(g_q, 1, 2), g_q)
    gTg_inv = torch.inverse(gTg)
    gTg_inv_gT = torch.matmul(gTg_inv, torch.transpose(g_q, 1, 2))
    u = torch.matmul(gTg_inv_gT, (dH_a - k_d * q_dot))
    u = u[0]
    u = u.detach().cpu().numpy()

    # Apply control u
    obs, _, _, _ = env.step(u)

    # Update state for next time step.
    y = np.concatenate((obs, np.array([u0])))
    y = torch.tensor(y, requires_grad=True, device=device, dtype=torch.float32).view(1, 13)

    # Save states for plotting
    s = env.get_state()
    s = np.concatenate((s, u[0]))
    state_traj.append(s)

# Plot trajectory
plot_traj(state_traj, t_eval)
env.close()