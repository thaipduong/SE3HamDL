# Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control, RSS 2021
# Thai Duong, Nikolay Atanasov

# code structure follows the style of HNN by Greydanus et al. and SymODEM by Zhong et al.
# https://github.com/greydanus/hamiltonian-nn
# https://github.com/Physics-aware-AI/Symplectic-ODENet

import torch, argparse
import numpy as np
import os, sys
from torchdiffeq import odeint_adjoint as odeint
from se3hamneuralode import MLP, PSD
from se3hamneuralode import SE3HamNODE
from data_collection import get_dataset, arrange_data
from se3hamneuralode import to_pickle, pose_L2_geodesic_loss, traj_pose_L2_geodesic_loss
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))+'/data'
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--learn_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=5000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=100, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='quadrotor', type=str, help='only one option right now')
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

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the model
    if args.verbose:
        print("Start training with num of points = {} and solver {}.".format(args.num_points, args.solver))
    model = SE3HamNODE(device=device, pretrain = False).to(device)
    # Load saved params if needed
    #path = '{}/quadrotor-se3ham-rk4-5p.tar'.format(args.save_dir)
    #model.load_state_dict(torch.load(path, map_location=device))
    # Save/load pre-init model
    #path = '{}/quadrotor-se3ham-rk4-5p-pre-init.tar'.format(args.save_dir)
    #model.load_state_dict(torch.load(path, map_location=device))
    #torch.save(model.state_dict(), path)
    num_parm = get_model_parm_nums(model)
    print('Model contains {} parameters'.format(num_parm))
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0.0)

    # Collect data
    data = get_dataset(test_split=0.8, save_dir=args.save_dir)
    train_x, t_eval = arrange_data(data['x'], data['t'], num_points=args.num_points)
    test_x, t_eval = arrange_data(data['test_x'], data['t'], num_points=args.num_points)
    train_x_cat = np.concatenate(train_x, axis=1)
    test_x_cat = np.concatenate(test_x, axis=1)
    train_x_cat = torch.tensor(train_x_cat, requires_grad=True, dtype=torch.float32).to(device)
    test_x_cat = torch.tensor(test_x_cat, requires_grad=True, dtype=torch.float32).to(device)
    #train_x = torch.tensor(train_x, requires_grad=True, dtype=torch.float32).to(device)
    #test_x = torch.tensor(test_x, requires_grad=True, dtype=torch.float32).to(device)
    t_eval = torch.tensor(t_eval, requires_grad=True, dtype=torch.float32).to(device)


    # Training stats
    stats = {'train_loss': [], 'test_loss': [], 'forward_time': [], 'backward_time': [], 'nfe': [], 'train_x_loss': [],\
             'test_x_loss':[], 'train_v_loss': [], 'test_v_loss': [], 'train_w_loss': [], 'test_w_loss': [], 'train_geo_loss':[], 'test_geo_loss':[]}

    # Start training
    for step in range(0,args.total_steps + 1):
        #print(step)
        train_loss = 0
        test_loss = 0
        train_x_loss = 0
        train_v_loss = 0
        train_w_loss = 0
        train_geo_loss = 0
        test_x_loss = 0
        test_v_loss = 0
        test_w_loss = 0
        test_geo_loss = 0

        t = time.time()
        # Predict states
        train_x_hat = odeint(model, train_x_cat[0, :, :], t_eval, method=args.solver)
        forward_time = time.time() - t
        target = train_x_cat[1:, :, :]
        target_hat = train_x_hat[1:, :, :]

        # Calculate loss
        train_loss_mini, x_loss_mini, v_loss_mini, w_loss_mini, geo_loss_mini = \
            pose_L2_geodesic_loss(target, target_hat, split=[model.xdim, model.Rdim, model.twistdim, model.udim])
        train_loss = train_loss + train_loss_mini
        train_x_loss = train_x_loss + x_loss_mini
        train_v_loss = train_v_loss + v_loss_mini
        train_w_loss = train_w_loss + w_loss_mini
        train_geo_loss = train_geo_loss + geo_loss_mini

        # Calculate loss for test data
        test_x_hat = odeint(model, test_x_cat[0, :, :], t_eval, method=args.solver)
        target = test_x_cat[1:, :, :]
        target_hat = test_x_hat[1:, :, :]
        test_loss_mini, x_loss_mini, v_loss_mini, w_loss_mini, geo_loss_mini = \
            pose_L2_geodesic_loss(target, target_hat, split=[model.xdim, model.Rdim, model.twistdim, model.udim])
        test_loss = test_loss + test_loss_mini
        test_x_loss = test_x_loss + x_loss_mini
        test_v_loss = test_v_loss + v_loss_mini
        test_w_loss = test_w_loss + w_loss_mini
        test_geo_loss = test_geo_loss + geo_loss_mini

        # Gradient descent
        t = time.time()
        if step > 0:
            train_loss_mini.backward()
            optim.step()
            optim.zero_grad()
        backward_time = time.time() - t

        # Logging stats
        stats['train_loss'].append(train_loss.item())
        stats['test_loss'].append(test_loss.item())
        stats['train_x_loss'].append(train_x_loss.item())
        stats['test_x_loss'].append(test_x_loss.item())
        stats['train_v_loss'].append(train_v_loss.item())
        stats['test_v_loss'].append(test_v_loss.item())
        stats['train_w_loss'].append(train_w_loss.item())
        stats['test_w_loss'].append(test_w_loss.item())
        stats['train_geo_loss'].append(train_geo_loss.item())
        stats['test_geo_loss'].append(test_geo_loss.item())
        stats['forward_time'].append(forward_time)
        stats['backward_time'].append(backward_time)
        stats['nfe'].append(model.nfe)
        if step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, train_loss.item(), test_loss.item()))
            print("step {}, train_x_loss {:.4e}, test_x_loss {:.4e}".format(step, train_x_loss.item(),
                                                                            test_x_loss.item()))
            print("step {}, train_v_loss {:.4e}, test_v_loss {:.4e}".format(step, train_v_loss.item(),
                                                                            test_v_loss.item()))
            print("step {}, train_w_loss {:.4e}, test_w_loss {:.4e}".format(step, train_w_loss.item(),
                                                                            test_w_loss.item()))
            print("step {}, train_geo_loss {:.4e}, test_geo_loss {:.4e}".format(step, train_geo_loss.item(),
                                                                                test_geo_loss.item()))
            print("step {}, nfe {:.4e}".format(step, model.nfe))
            # # Uncomment this to save model every args.print_every steps
            # os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
            # label = '-se3ham'
            # path = '{}/{}{}-{}-{}p3-{}.tar'.format(args.save_dir, args.name, label, args.solver, args.num_points, step)
            # torch.save(model.state_dict(), path)

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
        total_loss, l2_loss, geo_loss = \
            traj_pose_L2_geodesic_loss(train_x[i, :, :, :], train_x_hat, split=[model.xdim, model.Rdim, model.twistdim, model.udim])
        train_loss.append(total_loss)
        train_l2_loss.append(l2_loss)
        train_geo_loss.append(geo_loss)
        train_data_hat.append(train_x_hat.detach().cpu().numpy())

        # Run test data
        test_x_hat = odeint(model, test_x[i, 0, :, :], t_eval, method=args.solver)
        total_loss, l2_loss, geo_loss = \
            traj_pose_L2_geodesic_loss(test_x[i,:,:,:], test_x_hat, split=[model.xdim, model.Rdim, model.twistdim, model.udim])
        test_loss.append(total_loss)
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
    label = '-se3ham'
    path = '{}/{}{}-{}-{}p.tar'.format(args.save_dir, args.name, label, args.solver, args.num_points)
    torch.save(model.state_dict(), path)
    path = '{}/{}{}-{}-{}p-stats.pkl'.format(args.save_dir, args.name, label, args.solver, args.num_points)
    print("Saved file: ", path)
    to_pickle(stats, path)
