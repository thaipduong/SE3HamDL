# code structure follows the style of Symplectic ODE-Net
# https://github.com/d-biswa/Symplectic-ODENet/blob/master/experiment-single-embed/data.py

import numpy as np
from se3hamneuralode import to_pickle, from_pickle
import gym
import envs

def sample_gym(seed=0, timesteps=10, trials=50, min_angle=0., 
              verbose=False, u=0.0, env_name='MyPendulum-v1', ori_rep = 'angle',friction = False, render = False):
    
    gym_settings = locals()
    if verbose:
        print("Making a dataset of Pendulum observations.")
    env = gym.make(env_name)
    env.seed(seed)
    env.friction = friction
    trajs = []
    for trial in range(trials):
        valid = False
        while not valid:
            env.reset(ori_rep=ori_rep)
            traj = []
            for step in range(timesteps):
                if render:
                    env.render()
                obs, _, _, _ = env.step([u]) # action
                x = np.concatenate((obs, np.array([u])))
                traj.append(x)
            traj = np.stack(traj)
            if np.amax(traj[:, 2]) < env.max_speed - 0.001  and np.amin(traj[:, 2]) > -env.max_speed + 0.001:
                valid = True
        trajs.append(traj)
    trajs = np.stack(trajs) # (trials, timesteps, 2)
    trajs = np.transpose(trajs, (1, 0, 2)) # (timesteps, trails, 2)
    tspan = np.arange(timesteps) * 0.05
    return trajs, tspan, gym_settings


def get_dataset(seed=0, samples=50, test_split=0.9, save_dir=None, us=[0], rad=False, ori_rep = 'angle', friction = False, **kwargs):
    data = {}

    assert save_dir is not None
    path = '{}/pendulum-gym-dataset.pkl'.format(save_dir)
    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        trajs_force = []
        for u in us:
            trajs, tspan, _ = sample_gym(seed=seed, trials=samples, u=u, ori_rep = ori_rep, friction = friction, **kwargs)
            trajs_force.append(trajs)
        data['x'] = np.stack(trajs_force, axis=0) # (3, 45, 50, 3)
        # make a train/test split
        split_ix = int(samples * test_split)
        split_data = {}
        split_data['x'], split_data['test_x'] = data['x'][:,:,:split_ix,:], data['x'][:,:,split_ix:,:]

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

if __name__ == "__main__":
    #us = [0.0, -1.0, 1.0, -2.0, 2.0]
    us = [0.0]
    #data = get_dataset(seed=0, timesteps=20, save_dir=None, us=us, samples=128)
    trajs, tspan, _  = sample_gym(seed=0, trials=50, u=us[0], timesteps=20, ori_rep='6d')
    print("Done!")