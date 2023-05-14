# Modified from Symplectic-ODENet's Pendulum environment
# https://github.com/d-biswa/Symplectic-ODENet/blob/master/myenv/pendulum.py

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class PendulumEnvV1(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=10.0, ori_rep = 'angle', friction=False):
        self.max_speed=100.
        self.max_torque=5.
        self.dt=.05
        self.g = g
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Orientation representations: 'angle', 'rotmat'
        self.ori_rep = ori_rep
        self.friction = friction
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def dynamics(self, t, y, u):
        g = self.g
        m = 1.
        l = 1.
        friction = 0.3*y[1] if self.friction else 0.0
        f = np.zeros_like(y)
        f[0] = y[1]
        f[1] = (-3*g/(2*l) * np.sin(y[0]) + 3./(m*l**2)*u) - friction
        return f

    def get_state(self):
        return self.state

    def step(self,u):
        th, thdot = self.state # th := theta

        g = self.g
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        ivp = solve_ivp(fun=lambda t, y:self.dynamics(t, y, u), t_span=[0, self.dt], y0=self.state)
        self.state = ivp.y[:, -1]

        return self.get_obs(), -costs, False, {}

    def reset(self, ori_rep = 'angle', init_state = None):
        if init_state is None:
            high = np.array([np.pi, 1])
            self.state = self.np_random.uniform(low=-high, high=high)
        else:
            self.state = init_state
        self.last_u = None
        # Orientation representations: 'angle', 'rotmat'
        self.ori_rep = ori_rep
        return self.get_obs()

    def get_obs(self):
        theta, thetadot = self.state
        w = np.array([0.0, 0.0, thetadot])
        if self.ori_rep == 'angle':
            ret = np.array([theta, thetadot])
        if self.ori_rep == 'rotmat':
            R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                          [np.sin(theta),  np.cos(theta), 0.0],
                          [0.0,            0.0,           1.0]])
            ret = np.hstack((R.flatten(), w))
        return ret

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] - np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None