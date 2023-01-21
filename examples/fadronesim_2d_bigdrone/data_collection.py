# code structure follows the style of Symplectic ODE-Net
# https://github.com/d-biswa/Symplectic-ODENet/blob/master/experiment-single-embed/data.py

import time
import argparse
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from se3hamneuralode import to_pickle, from_pickle
from envs.laser_sim import *
from envs.fadronesim import *
from controllers.controller_pid import *
from controllers.controller_utils import *

# Pybullet drone environment
#from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
#from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
#from gym_pybullet_drones.utils.utils import sync, str2bool
# from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

parser = argparse.ArgumentParser(description='Debugging script for PyBullet applyExternalForce() and applyExternalTorque() PyBullet')
parser.add_argument('--duration_sec',   default=1.0,     type=float,       help='Duration of the simulation in seconds (default: 30)', metavar='')
parser.add_argument('--num_resets',     default=1,      type=int,       help='Number of times the simulation is reset to its initial conditions (default: 2)', metavar='')
parser.add_argument('--simulation_freq_hz', default=240, type=int, help='Simulation frequency in Hz (default: 240)', metavar='')
ARGS = parser.parse_args()



def sample_gym():
    gpmpc_data = np.load("./data/statecontroldata_rand_120Hz.npz")
    # current state
    x_seq_all = gpmpc_data["x_seq_all"][:,0,:]
    # control input
    u_seq_all = gpmpc_data["u_seq_all"][:, 0, :]
    # next state
    x_next_seq_all = gpmpc_data["x_next_seq_all"][:,0,:]

    data_set = np.zeros((1, 2, x_seq_all.shape[0], 21))
    for i in range(x_seq_all.shape[0]):
        R = Rotation.from_euler("zyx", [0., x_seq_all[i,4], 0.])
        rotmat = R.as_matrix()
        origin = np.array([0., 0., 0.])
        pos = np.array([x_seq_all[i,0], 0., x_seq_all[i,2]])
        v_bodyframe = np.matmul(rotmat.T, np.array([x_seq_all[i,1], 0., x_seq_all[i,3]]))
        w_bodyframe = np.array([0., x_seq_all[i,5], 0.])
        curr_state = np.concatenate((origin, rotmat.flatten(), v_bodyframe, w_bodyframe, u_seq_all[i,:]))
        data_set[0, 0, i, :] = curr_state
        R = Rotation.from_euler("zyx", [0., x_next_seq_all[i,4], 0.])
        rotmat = R.as_matrix()
        pos = np.array([x_next_seq_all[i,0], 0., x_next_seq_all[i,2]]) - pos
        v_bodyframe = np.matmul(rotmat.T, np.array([x_next_seq_all[i,1], 0., x_next_seq_all[i,3]]))
        w_bodyframe = np.array([0., x_next_seq_all[i,5], 0.])
        next_state = np.concatenate((pos, rotmat.flatten(), v_bodyframe, w_bodyframe, u_seq_all[i,:]))
        data_set[0, 1, i, :] = next_state

    tspan = np.arange(2)/120

    return data_set, None, tspan

def get_dataset(test_split=0.5, save_dir=None, **kwargs):
    data = {}

    assert save_dir is not None
    path = '{}/hexarotor_2D_gpmpc_offset_120Hz.pkl'.format(save_dir)
    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        data_set,_, tspan = sample_gym()
        # Make a train/test split
        samples = data_set.shape[2]
        split_ix = int(samples * test_split)
        split_data = {}
        split_data['x'], split_data['test_x'] = data_set[:,:,:split_ix,:], data_set[:,:,split_ix:,:]
        data = split_data
        data['t'] = tspan
        to_pickle(data, path)
    return data


def arrange_data(x, t, num_points=2):
    '''Arrange data to feed into neural ODE in small chunks'''
    assert num_points>=2 and num_points<=len(t)
    x_stack = []
    for i in range(num_points):
        if i < num_points-1:
            x_stack.append(x[:, i:-num_points+i+1,:,:])
        else:
            x_stack.append(x[:, i:,:,:])
    x_stack = np.stack(x_stack, axis=1)
    x_stack = np.reshape(x_stack,
                (x.shape[0], num_points, -1, x.shape[3]))
    t_eval = t[0:num_points]
    return x_stack, t_eval

# if __name__ == "__main__":
#     data_set, trajectories, tspan = sample_gym()
#     for i in range(len(trajectories)):
#         plot_traj1D(trajectories[i])