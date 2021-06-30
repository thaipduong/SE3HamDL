# Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control, RSS 2021
# Thai Duong, Nikolay Atanasov

# code structure follows the style of HNN by Greydanus et al. and SymODEM by Zhong et al.
# https://github.com/greydanus/hamiltonian-nn
# https://github.com/Physics-aware-AI/Symplectic-ODENet

import torch, argparse
import numpy as np
import os, sys
import time
from torchdiffeq import odeint_adjoint as odeint

THIS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from se3hamneuralode import MLP, PSD
from se3hamneuralode import SO3HamNODE
from data import get_dataset, arrange_data
from se3hamneuralode import to_pickle, rotmat_L2_geodesic_loss, traj_rotmat_L2_geodesic_loss

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--learn_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=200, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=20, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='pendulum', type=str, help='environment name')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_points', type=int, default=5,
                        help='number of evaluation points by the ODE solver, including the initial point')
    parser.add_argument('--solver', default='rk4', type=str, help='type of ODE Solver for Neural ODE')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def get_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total


def train(args):

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # reproducibility: set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the model
    if args.verbose:
        print("Start training with num of points = {} and solver {}.".format(args.num_points, args.solver))
    model = SO3HamNODE(device=device, u_dim = 1).to(device)
    num_parm = get_model_parm_nums(model)
    print('model contains {} parameters'.format(num_parm))
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

    # Collect data
    us = [0.0, -1.0, 1.0, -2.0, 2.0]
    data = get_dataset(seed=args.seed, timesteps=20, save_dir=args.save_dir, us=us, ori_rep="rotmat", samples=64)
    train_x, t_eval = arrange_data(data['x'], data['t'], num_points=args.num_points)
    test_x, t_eval = arrange_data(data['test_x'], data['t'], num_points=args.num_points)
    train_x = torch.tensor(train_x, requires_grad=True, dtype=torch.float32).to(device)
    test_x = torch.tensor(test_x, requires_grad=True, dtype=torch.float32).to(device)
    t_eval = torch.tensor(t_eval, requires_grad=True, dtype=torch.float32).to(device)

    # Training stats
    stats = {'train_loss': [], 'test_loss': [], 'forward_time': [], 'backward_time': [], 'nfe': [], 'train_l2_loss': [],\
             'test_l2_loss':[], 'train_geo_loss':[], 'test_geo_loss':[]}

    # Start training
    for step in range(args.total_steps + 1):
        train_loss = 0
        test_loss = 0
        train_l2_loss = 0
        train_geo_loss = 0
        test_l2_loss = 0
        test_geo_loss = 0
        for i in range(train_x.shape[0]):
            t = time.time()
            # Predict states
            train_x_hat = odeint(model, train_x[i, 0, :, :], t_eval, method=args.solver)
            forward_time = time.time() - t
            target = train_x[i, 1, :, :]
            target_hat = train_x_hat[1, :, :]
            # Calculate loss
            train_loss_mini, l2_loss_mini, geo_loss_mini = rotmat_L2_geodesic_loss(target, target_hat, split=[model.rotmatdim, model.angveldim, 1]) #L2_loss(target, target_hat)#
            train_loss = train_loss + train_loss_mini
            train_l2_loss = train_l2_loss + l2_loss_mini
            train_geo_loss = train_geo_loss + geo_loss_mini

            # Gradient descent
            train_loss_mini.backward()
            optim.step()
            optim.zero_grad()
            backward_time = time.time() - t

            # Calculate loss for test data
            test_x_hat = odeint(model, test_x[i, 0, :, :], t_eval, method=args.solver)
            target = test_x[i, 1, :, :]
            target_hat = test_x_hat[1, :, :]
            test_loss_mini, l2_loss_mini, geo_loss_mini = rotmat_L2_geodesic_loss(target, target_hat, split=[model.rotmatdim, model.angveldim, 1])
            test_loss = test_loss + test_loss_mini
            test_l2_loss = test_l2_loss + l2_loss_mini
            test_geo_loss = test_geo_loss + geo_loss_mini

        # Logging stats
        stats['train_loss'].append(train_loss.item())
        stats['test_loss'].append(test_loss.item())
        stats['train_l2_loss'].append(train_l2_loss.item())
        stats['test_l2_loss'].append(test_l2_loss.item())
        stats['train_geo_loss'].append(train_geo_loss.item())
        stats['test_geo_loss'].append(test_geo_loss.item())
        stats['forward_time'].append(forward_time)
        stats['backward_time'].append(backward_time)
        stats['nfe'].append(model.nfe)
        if step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, train_loss.item(), test_loss.item()))
            print("step {}, train_l2_loss {:.4e}, test_l2_loss {:.4e}".format(step, train_l2_loss.item(),
                                                                              test_l2_loss.item()))
            print("step {}, train_geo_loss {:.4e}, test_geo_loss {:.4e}".format(step, train_geo_loss.item(),
                                                                                test_geo_loss.item()))
            print("step {}, nfe {:.4e}".format(step, model.nfe))

    # Calculate loss mean and standard deviation
    train_x, t_eval = data['x'], data['t']
    test_x, t_eval = data['test_x'], data['t']

    train_x = torch.tensor(train_x, requires_grad=True, dtype=torch.float32).to(device)
    test_x = torch.tensor(test_x, requires_grad=True, dtype=torch.float32).to(device)
    t_eval = torch.tensor(t_eval, requires_grad=True, dtype=torch.float32).to(device)

    train_loss = []
    test_loss = []
    train_l2_loss = []
    test_l2_loss = []
    train_geo_loss = []
    test_geo_loss = []
    train_data_hat = []
    test_data_hat = []
    for i in range(train_x.shape[0]):
        train_x_hat = odeint(model, train_x[i, 0, :, :], t_eval, method=args.solver)
        total_loss, l2_loss, geo_loss = traj_rotmat_L2_geodesic_loss(train_x[i, :, :, :], train_x_hat, split=[model.rotmatdim, model.angveldim, 1])
        train_loss.append(total_loss)#((train_x[i,:,:,:] - train_x_hat)**2)#(total_loss)#
        train_l2_loss.append(l2_loss)
        train_geo_loss.append(geo_loss)
        train_data_hat.append(train_x_hat.detach().cpu().numpy())

        # run test data
        test_x_hat = odeint(model, test_x[i, 0, :, :], t_eval, method=args.solver)
        total_loss, l2_loss, geo_loss = traj_rotmat_L2_geodesic_loss(test_x[i,:,:,:], test_x_hat, split=[model.rotmatdim, model.angveldim, 1])
        test_loss.append(total_loss)#((test_x[i,:,:,:] - test_x_hat)**2)#(total_loss)#((test_x[i,:,:,:] - test_x_hat)**2)
        test_l2_loss.append(l2_loss)
        test_geo_loss.append(geo_loss)
        test_data_hat.append(test_x_hat.detach().cpu().numpy())

    train_loss = torch.cat(train_loss, dim=1)
    train_loss_per_traj = torch.sum(train_loss, dim=0)

    test_loss = torch.cat(test_loss, dim=1)
    test_loss_per_traj = torch.sum(test_loss, dim=0)

    train_l2_loss = torch.cat(train_l2_loss, dim=1)
    train_l2_loss_per_traj = torch.sum(train_l2_loss, dim=0)

    test_l2_loss = torch.cat(test_l2_loss, dim=1)
    test_l2_loss_per_traj = torch.sum(test_l2_loss, dim=0)

    train_geo_loss = torch.cat(train_geo_loss, dim=1)
    train_geo_loss_per_traj = torch.sum(train_geo_loss, dim=0)

    test_geo_loss = torch.cat(test_geo_loss, dim=1)
    test_geo_loss_per_traj = torch.sum(test_geo_loss, dim=0)

    print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
    .format(train_loss_per_traj.mean().item(), train_loss_per_traj.std().item(),
            test_loss_per_traj.mean().item(), test_loss_per_traj.std().item()))
    print('Final trajectory train l2 loss {:.4e} +/- {:.4e}\nFinal trajectory test l2 loss {:.4e} +/- {:.4e}'
    .format(train_l2_loss_per_traj.mean().item(), train_l2_loss_per_traj.std().item(),
            test_l2_loss_per_traj.mean().item(), test_l2_loss_per_traj.std().item()))
    print('Final trajectory train geo loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
    .format(train_geo_loss_per_traj.mean().item(), train_geo_loss_per_traj.std().item(),
            test_geo_loss_per_traj.mean().item(), test_geo_loss_per_traj.std().item()))

    #train_data_hat = torch.cat(train_data_hat, dim=0)

    stats['traj_train_loss'] = train_loss_per_traj.detach().cpu().numpy()
    stats['traj_test_loss'] = test_loss_per_traj.detach().cpu().numpy()
    stats['train_x'] = train_x.detach().cpu().numpy()
    stats['test_x'] = test_x.detach().cpu().numpy()
    stats['train_x_hat'] = np.array(train_data_hat)
    stats['test_x_hat'] = np.array(test_data_hat)
    stats['t_eval'] = t_eval.detach().cpu().numpy()
    return model, stats


if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # Save model
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-so3ham_ode'
    struct = '-struct'
    path = '{}/{}{}-{}-{}p.tar'.format(args.save_dir, args.name, label, args.solver, args.num_points)
    torch.save(model.state_dict(), path)
    path = '{}/{}{}-{}-{}p-stats.pkl'.format(args.save_dir, args.name, label, args.solver,
                                                    args.num_points)
    print("saved file: ", path)
    to_pickle(stats, path)
