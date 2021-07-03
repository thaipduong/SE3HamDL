import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from se3hamneuralode import compute_rotation_matrix_from_quaternion, from_pickle, SO3HamNODE
solve_ivp = scipy.integrate.solve_ivp

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = True

gpu=0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

def plot_traj(traj, traj_hat, t_eval):
    ''' Plot trajectory '''
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
    model = SO3HamNODE(device=device, u_dim=1).to(device)
    path = './data/pendulum-so3ham-rk4-5p.tar'
    model.load_state_dict(torch.load(path, map_location=device))
    path = './data/pendulum-so3ham-rk4-5p-stats.pkl'
    stats = from_pickle(path)
    return model, stats

if __name__ == "__main__":
    # Figure and font size
    figsize = (12, 7.8)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 4
    # Load trained model
    model, stats = get_model()
    # Scale factor for M^-1, V, g neural networks
    beta = 5.1

    # Load train/test data
    train_x_hat = stats['train_x_hat']
    test_x_hat = stats['test_x_hat']
    train_x = stats['train_x']
    test_x = stats['test_x']
    t_eval = stats['t_eval']
    print("Loaded data!")

    # Plot loss
    fig = plt.figure(figsize=figsize, linewidth=5)
    ax = fig.add_subplot(111)
    train_loss = stats['train_loss']
    test_loss = stats['test_loss']
    ax.plot(train_loss[0:], 'b', linewidth=line_width, label='train loss')
    ax.plot(test_loss[0:], 'r', linewidth=line_width, label='test loss')
    plt.xlabel("iterations", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.yscale('log')
    plt.legend(fontsize=fontsize)
    plt.savefig('./png/loss_log.png', bbox_inches='tight')
    plt.show()

    # Get state q from a range of pendulum angle theta
    theta = np.linspace(-5.0, 5.0, 40)
    q_tensor = torch.tensor(theta, dtype=torch.float32).view(40, 1).to(device)
    q_zeros = torch.zeros(40,2).to(device)
    quat = torch.cat((torch.cos(q_tensor/2), q_zeros, torch.sin(q_tensor/2)), dim=1)
    rotmat = compute_rotation_matrix_from_quaternion(quat)
    # This is the generalized coordinates q = R
    rotmat = rotmat.view(rotmat.shape[0], 9)



    # Calculate the M^-1, V, g for the q.
    M_q_inv = model.M_net(rotmat)
    V_q = model.V_net(rotmat)
    g_q = model.g_net(rotmat)

    # Plot g(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(theta, beta*g_q.detach().cpu().numpy()[:,0], 'b--', linewidth=line_width, label=r'$\beta g(q)[1]$')
    plt.plot(theta, beta * g_q.detach().cpu().numpy()[:, 1], 'r--', linewidth=line_width, label=r'$\beta g(q)[2]$')
    plt.plot(theta, beta * g_q.detach().cpu().numpy()[:, 2], 'g--', linewidth=line_width, label=r'$\beta g(q)[3]$')
    plt.xlabel("pendulum angle", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlim(-5, 5)
    plt.ylim(-0.5, 2.5)
    plt.legend(fontsize=fontsize)
    plt.savefig('./png/g_x.png', bbox_inches='tight')
    plt.show()

    # Plot V(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(theta, 5. - 5. * np.cos(theta), 'k--', label='Ground Truth', color='k', linewidth=line_width)
    plt.plot(theta, beta*V_q.detach().cpu().numpy(), 'b', label=r'$\beta V(q)$', linewidth=line_width)
    plt.xlabel("pendulum angle", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlim(-5, 5)
    plt.ylim(-8, 12)
    plt.legend(fontsize=fontsize)
    plt.savefig('./png/V_x.png', bbox_inches='tight')
    plt.show()

    # Plot M^-1(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(theta, 3 * np.ones_like(theta), label='Ground Truth', color='k', linewidth=line_width-1)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 2, 2] / beta, 'b--', linewidth=line_width,
             label=r'$M^{-1}(q)[3, 3]/\beta$')
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 0, 0] / beta, 'g--', linewidth=line_width,
             label=r'Other $M^{-1}(q)[i,j]/\beta$')
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 0, 1] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 0, 2] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 1, 0] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 1, 1] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 1, 2] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 2, 0] / beta, 'g--', linewidth=line_width)
    plt.plot(theta, M_q_inv.detach().cpu().numpy()[:, 2, 1] / beta, 'g--', linewidth=line_width)
    plt.xlabel("pendulum angle", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlim(-5, 5)
    plt.ylim(-0.5, 6.0)
    plt.legend(fontsize=fontsize)
    plt.savefig('./png/M_x_all.png', bbox_inches='tight')
    plt.show()
