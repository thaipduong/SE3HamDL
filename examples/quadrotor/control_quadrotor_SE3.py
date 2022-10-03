import time
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from se3hamneuralode import to_pickle, from_pickle

# import from pybullet drone environment
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from controller_energy_based import *

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = True

# pybullet drone parameters
parser = argparse.ArgumentParser(description='Debugging script for PyBullet applyExternalForce() and applyExternalTorque() PyBullet')
parser.add_argument('--duration_sec',   default=20,     type=float,       help='Duration of the simulation in seconds (default: 30)', metavar='')
parser.add_argument('--num_resets',     default=1,      type=int,       help='Number of times the simulation is reset to its initial conditions (default: 2)', metavar='')
parser.add_argument('--simulation_freq_hz', default=240, type=int, help='Simulation frequency in Hz (default: 240)', metavar='')
parser.add_argument('--control_freq_hz', default=240, type=int, help='Control frequency in Hz (default: 48)', metavar='')
ARGS = parser.parse_args()
AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz)

# Start demo
def control_pybullet_drone_demo():
    # Instantiate an energy-based controller using the learned Hamiltonian dynamics on SE(3)
    learned_energy_based_controller = LearnedEnergyBasedController()

    # Initial position and orientation of the drone
    INIT_XYZS = np.array([0.0, 0.0, 0.0]).reshape(1, 3)
    INIT_RPYS = np.array([0, 0, 30 * (np.pi / 180)]).reshape(1, 3)

    # Get desired trajectory (diamond-shaped)
    h_traj = diamond
    s_plan_start = h_traj(0)
    t_start = 0  # start of simulation in seconds
    t_step = 1/ARGS.control_freq_hz

    # Start to track the desired state for plotting
    s_plan = []
    s_traj = []
    curr_plan_s = np.zeros(11)
    curr_plan_s[0:3] = s_plan_start.pos + INIT_XYZS.flatten()
    curr_plan_s[3:6] = s_plan_start.vel
    curr_plan_s[6:9] = s_plan_start.acc
    curr_plan_s[9] = s_plan_start.yaw
    curr_plan_s[-1] = t_start
    s_plan.append(curr_plan_s)

    # Start pybullet drone environment
    env = CtrlAviary(drone_model=DroneModel.CF2P,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=Physics.PYB, #physics=Physics.PYB_DRAG,
                     gui=True,
                     record = False,
                     obstacles=True,
                     user_debug_gui=False
                     )

    # Get PyBullet's and drone's ids
    PYB_CLIENT = env.getPyBulletClient()
    DRONE_IDS = env.getDroneIds()

    # Draw base frame on the drone
    env._showDroneLocalAxes(0)

    # Start the simulation with no control
    action = {'0': np.array([0,0,0,0])}
    START = time.time()
    obs, reward, done, info = env.step(action)
    state = obs['0']["state"]
    quat = state[3:7]
    R = Rotation.from_quat(quat)
    rotmat = R.as_matrix()
    # Save current drone state for plotting
    t_curr = 0.0
    w_bodyframe = np.matmul(rotmat.T, state[13:16])
    curr_traj_s = np.zeros(14)
    curr_traj_s[0:3] = state[0:3]
    curr_traj_s[3:6] = state[10:13]
    curr_traj_s[6:9] = np.array([0,0,0])
    curr_traj_s[9] = state[9]
    curr_traj_s[10:13] = w_bodyframe[:]
    curr_traj_s[-1] = t_curr
    s_traj.append(curr_traj_s)

    # Build conversion matrix betweek force/torch and the motor speeds.
    # We need this because the pybullet drone environment takes motor speeds as input,
    # while in our setting, the control input is force/torch.
    r = env.KM / env.KF
    conversion_mat = np.array([[1.0, 1.0, 1.0, 1.0],
                               [0.0, env.L, 0.0, -env.L],
                               [-env.L, 0.0, env.L, 0.0],
                               [-r, r, -r, r]])
    conversion_mat_inv = np.linalg.inv(conversion_mat)

    # Start trajectory tracking for 20 seconds
    # The controller is called at 240 Hz.
    recorded_time = 0
    for i in range(int(ARGS.duration_sec*ARGS.control_freq_hz)):
        # Desired state
        desired_state = h_traj(t_curr + t_step)
        # Save the desired state for plotting
        curr_plan_s = np.zeros(11)
        curr_plan_s[0:3] = desired_state.pos + INIT_XYZS.flatten()
        curr_plan_s[3:6] = desired_state.vel
        curr_plan_s[6:9] = desired_state.acc
        curr_plan_s[9] = desired_state.yaw
        curr_plan_s[-1] = t_curr
        s_plan.append(curr_plan_s)
        # Save the current drone state for plotting
        state = obs['0']["state"]
        quat = state[3:7]
        R = Rotation.from_quat(quat)
        rotmat = R.as_matrix()
        w_bodyframe = np.matmul(rotmat.T, state[13:16])
        curr_traj_s = np.zeros(14)
        curr_traj_s[0:3] = state[0:3]
        curr_traj_s[3:6] = state[10:13]
        curr_traj_s[6:9] = np.array([0, 0, 0])
        curr_traj_s[9] = state[9]
        curr_traj_s[10:13] = w_bodyframe[:]
        curr_traj_s[-1] = t_curr
        s_traj.append(curr_traj_s)

        # Call our energy-based controller using the learned dynamics
        #### Assemble the control state and pass it to the controller
        a = time.time()
        s = np.zeros(13)
        s[0] = state[0]  # x
        s[1] = state[1]  # y
        s[2] = state[2]  # z
        s[3] = state[10]  # xdot
        s[4] = state[11]  # ydot
        s[5] = state[12]  # zdot
        s[6] = state[6]  # qw
        s[7] = state[3]  # qx
        s[8] = state[4]  # qy
        s[9] = state[5]  # qz
        s[10] = w_bodyframe[0]  # p
        s[11] = w_bodyframe[1]  # q
        s[12] = w_bodyframe[2]  # r
        qd = stateToQd(s)
        qd.pos_des = desired_state.pos + INIT_XYZS[0, :]
        qd.vel_des = desired_state.vel
        qd.acc_des = desired_state.acc
        qd.yaw_des = desired_state.yaw
        qd.yawdot_des = desired_state.yawdot

        #### Call our energy-based controller using the learned dynamics
        [thrust, torques] = learned_energy_based_controller.gen_control(qd, t_curr)  # controller_lee(qd, t_curr, params)
        thrust_torques = np.array([thrust, torques[0], torques[1], torques[2]])
        b = time.time()
        print("Controller time: ", b - a)

        # Convert thrust/torque to motor speeds
        rpm_squared = np.matmul(conversion_mat_inv, thrust_torques)
        rpm_squared[0] = max(0, rpm_squared[0])
        rpm_squared[1] = max(0, rpm_squared[1])
        rpm_squared[2] = max(0, rpm_squared[2])
        rpm_squared[3] = max(0, rpm_squared[3])
        rpm_from_thrusttorques = np.sqrt(rpm_squared / env.KF)
        action['0'] = rpm_from_thrusttorques

        # Apply the control (motor speed)
        for j in range(AGGR_PHY_STEPS):
            obs, reward, done, info = env.step(action)
            recorded_time = recorded_time + 1/env.SIM_FREQ
            env.render()

        sync(i, START, env.TIMESTEP*AGGR_PHY_STEPS)
        t_curr = t_curr + 1/ARGS.control_freq_hz

    # Close the environment
    env.close()
    s_traj = np.array(s_traj)
    s_plan = np.array(s_plan)
    plot_states1D(s_traj, s_plan)
    quadplot_update(s_traj, s_plan)

    for i in range(0,len(s_traj)-5,5):
        plot_states1D_i(s_traj, s_plan, i+1)
        quadplot_update_video(s_traj, s_plan, None, i+1)
    #np.savez("./data/traj_plan.npz", s_plan=s_plan, s_traj=s_traj)
    return

if __name__ == '__main__':
    control_pybullet_drone_demo()