import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from scipy.spatial.transform import Rotation
from scipy.linalg import expm
from scipy.constants import g
import matplotlib.pyplot as plt
from  scipy import integrate
solve_ivp = integrate.solve_ivp
import envs.runge_kutta_4th

class Quadrotor(gym.Env):
    def __init__(self, sim_freq = 120, init_xyz = None, init_rpys = None, init_vel = None, init_omega = None, render_flag = False):
        # Constants
        # Mass and inertia copied from crazy flie model
        # https://github.com/JacopoPan/gym-pybullet-drones/blob/master/gym_pybullet_drones/assets/cf2x.urdf
        self.M = 6.77 #0.1 #
        self.J = np.diag([1.05, 1.05, 2.05]) #0.25*np.diag([1.0, 1.0, 2.0])#
        self.J_inv = np.linalg.inv(self.J)
        self.G = 9.8
        self.friction = False
        #self.K_friction = np.diag([0.01, 0.01, 0.01])
        self.e3 = np.array([0.0, 0.0, 1.0])
        self.THRUST2WEIGHT_RATIO = 2.25
        self.KF = 3.16e-6
        self.KM = 7.94e-8
        self.L = 0.0397
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.M*self.G) / (4 * self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)

        # States include: x, x_dot, quat, R.
        if init_xyz is None:
            init_xyz = np.array([0.0, 0.0, 0.0])
        if init_rpys is None:
            init_rpys = np.array([0.0, 0.0, 0.0])
        if init_vel is None:
            init_vel = np.zeros(3)
        if init_omega is None:
            init_omega = np.zeros(3)
        self.x = init_xyz
        self.x_dot = init_vel#np.zeros(3)
        #self.x_dot_dot = np.zeros(3)
        rotation = Rotation.from_euler('xyz', init_rpys)
        self.rpys = init_rpys
        self.R = rotation.as_matrix()
        self.quat = rotation.as_quat()
        self.omega = init_omega#np.array([0.0, 0.0, 0.0])
        #self.omega_dot = np.array([0.0, 0.0, 0.0])

        # Time step
        self.dt = 1/sim_freq

        # Output is rot mat or pose?
        #self.manifold = manifold

        # Action space
        act_lower_bound = np.array([-self.MAX_THRUST, -self.MAX_XY_TORQUE, -self.MAX_XY_TORQUE, -self.MAX_Z_TORQUE])
        act_upper_bound = np.array([self.MAX_THRUST, self.MAX_XY_TORQUE, self.MAX_XY_TORQUE, self.MAX_Z_TORQUE])
        self.action_space = spaces.Box(low=act_lower_bound, high=act_upper_bound)

        # Init function
        self.seed()

        # Render
        self.render_flag = render_flag
        self.reset_viz()
        self.all_obs = np.expand_dims(self._get_obs(), axis=0)
        self.all_t = [0.0]
        self.curr_t = 0.0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def hat_map(self, w):
        w_hat = np.array([[0, -w[2], w[1]],
                          [w[2], 0, -w[0]],
                          [-w[1], w[0], 0]])
        return w_hat

    def dynamics(self, t, y, u):
        f = u[0]*self.e3
        #f = np.clip(f, -self.MAX_THRUST, self.MAX_THRUST)
        tau = u[1:4]
        #tau[0:2] = np.clip(tau[0:2], -self.MAX_XY_TORQUE, self.MAX_XY_TORQUE)
        #tau[2] = np.clip(tau[2], -self.MAX_Z_TORQUE, self.MAX_Z_TORQUE)
        x = y[0:3]
        R = y[3:12].reshape([3,3])
        x_dot = y[12:15]
        omega = y[15:18]
        x_dot_dot = np.matmul(self.R, f) / self.M - self.G * self.e3
        omega_dot = np.matmul(self.J_inv, tau - np.cross(self.omega, np.matmul(self.J, self.omega)))
        R_dot = np.matmul(R, self.hat_map(omega)).flatten()
        y_dot = np.hstack((x_dot, R_dot, x_dot_dot, omega_dot))
        return y_dot


    def step(self, action):
        #f = action[0]*self.e3
        #f = np.clip(f, -self.MAX_THRUST, self.MAX_THRUST)
        #tau = action[1:4]
        #tau[0:2] = np.clip(tau[0:2], -self.MAX_XY_TORQUE, self.MAX_XY_TORQUE)
        #tau[2] = np.clip(tau[2], -self.MAX_Z_TORQUE, self.MAX_Z_TORQUE)
        #steps = 50
        #print("Friction:", self.friction)
        y0 = self._get_rk45_state()
        ivp = solve_ivp(fun=lambda t, y: self.dynamics(t, y, action), t_span=[0, self.dt], y0=y0)
        #rk4_zhl = runge_kutta_4th.rk4_update(self.dt, self.dynamics, self.curr_t, y0, action)
        y = ivp.y[:, -1]
        #print(y - rk4_zhl)
        self.x = y[0:3]
        self.R = self.normalize_rotmat(y[3:12])
        rotation = Rotation.from_matrix(self.R)
        self.rpys = rotation.as_euler('xyz')
        self.quat = rotation.as_quat()
        self.x_dot = y[12:15]
        self.omega = y[15:18]
        #self.x_dot_bodyframe = np.matmul(np.transpose(self.R), self.x_dot)

        # for i in range(steps):
        #     euler_dt = self.dt/steps
        #     #friction = np.matmul(self.K_friction, self.x_dot_bodyframe)*(1.0 if self.friction else 0.0)
        #     #print("friction", friction)
        #     self.x_dot_dot = np.matmul(self.R, f)/self.M - self.G*self.e3
        #     self.x = self.x + euler_dt * self.x_dot + 0.5 * self.x_dot_dot * (euler_dt**2)
        #     self.x_dot = self.x_dot + euler_dt * self.x_dot_dot
        #     self.omega_dot = np.matmul(self.I_inv, tau - np.cross(self.omega, np.matmul(self.I, self.omega)))#np.array([0.0, 0.0, 0.0])#
        #     self.R = np.matmul(self.R, expm(euler_dt * self.hat_map(self.omega)))
        #     self.omega = self.omega + euler_dt*self.omega_dot


        #r = Rotation.from_matrix(self.R)
        #self.rpy = r.as_euler('xyz')
        #self.quat = r.as_quat()
        done = False
        if self.x[2] <=0:
            done = True
        obs = self._get_obs()
        self.all_obs = np.vstack((self.all_obs, obs))
        self.curr_t = self.curr_t + self.dt
        self.all_t.append(self.curr_t)
        return obs, -1, done, None

    def normalize_rotmat(self, unnormalized_rotmat):
        x_raw = unnormalized_rotmat[0:3]  # batch*3
        y_raw = unnormalized_rotmat[3:6]  # batch*3

        x = x_raw/np.linalg.norm(x_raw)
        z = np.cross(x, y_raw)  # batch*3
        z = z/np.linalg.norm(z)  # batch*3
        y = np.cross(z, x)  # batch*3

        matrix = np.vstack((x, y, z))  # batch*3*3
        # matrix = torch.transpose(matrix, 1, 2)
        return matrix

    def _get_rk45_state(self):
        return np.hstack((self.x, self.R.flatten(), self.x_dot, self.omega))

    def _get_obs(self):

        return np.hstack((self.x, self.quat, self.rpys, self.x_dot, self.omega, self.R.flatten()))

    def reset(self, init_xyz = None, init_rpys = None):
        # Reset state
        if init_xyz is None:
            init_xyz = np.array([0.0, 0.0, 0.0])
        if init_rpys is None:
            init_rpys = np.array([0.0, 0.0, 0.0])
        self.x = init_xyz
        self.x_dot = np.zeros(3)
        # self.x_dot_dot = np.zeros(3)
        rotation = Rotation.from_euler('xyz', init_rpys)
        self.rpys = init_rpys
        self.R = rotation.as_matrix()
        self.quat = rotation.as_quat()
        self.omega = np.array([0.0, 0.0, 0.0])

        # Reset viz
        self.reset_viz()
        self.all_obs = np.expand_dims(self._get_obs(), axis=0)
        self.all_t = [0.0]
        self.curr_t = 0.0
        return self._get_obs()

    def reset_viz(self):
        if self.render_flag:
            plt.close('all')
            self.fig = plt.figure(figsize=(12, 8), facecolor='white')
            self.subplot = []
            self.subplot.append(self.fig.add_subplot(331)) # , frameon=False)
            self.subplot.append(self.fig.add_subplot(332))  # , frameon=False)
            self.subplot.append(self.fig.add_subplot(333))  # , frameon=False)
            self.subplot.append(self.fig.add_subplot(334))  # , frameon=False)
            self.subplot.append(self.fig.add_subplot(335))  # , frameon=False)
            self.subplot.append(self.fig.add_subplot(336))  # , frameon=False)
            self.subplot.append(self.fig.add_subplot(337))  # , frameon=False)
            self.subplot.append(self.fig.add_subplot(338))  # , frameon=False)
            self.subplot.append(self.fig.add_subplot(339))  # , frameon=False)
            plt.show(block=False)

    def render(self, mode='human'):
        if self.render_flag:
            ylabel = ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot', 'roll', 'pitch', 'yaw']

            for i in range(len(self.subplot)):
                self.subplot[i].cla()
                self.subplot[i].set_title(ylabel[i])
                self.subplot[i].set_xlabel('t')
                self.subplot[i].set_ylabel(ylabel[i])
                self.subplot[i].plot(self.all_t, self.all_obs[:,i], 'g-')

            self.fig.tight_layout()
            #plt.savefig('png/{:05d}'.format(itr))
            plt.draw()
            plt.pause(0.001)



if __name__ == "__main__":
    quad1 = Quadrotor()
    quad1.reset()
    obs = quad1._get_obs()
    action = quad1.action_space.sample()
    quad1.step(action)
    obs = quad1._get_obs()
    action = quad1.action_space.sample()
    quad1.step(action)
    obs = quad1._get_obs()
    print("Applied control!")
