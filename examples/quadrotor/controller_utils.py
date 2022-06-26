# This is modified from Upenn MEAM 620 course:
# https://alliance.seas.upenn.edu/~meam620/wiki/index.php
import numpy as np
import torch
import matplotlib.pyplot as plt

class qd_object:
    """
    Struct to hold qd information
    """
    def __init__(self):
        self.pos = 0
        self.vel = 0
        self.euler = 0
        self.omega = 0

class state_object:
    """
    Struct to hold state information
    """
    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.yaw = 0
        self.yawdot = 0

def init_state(s_start):
    """
    Initialize 13 x 1 state vector
    """
    s     = np.zeros(13)
    phi0   = 0.0
    theta0 = 0.0
    psi0   = s_start.yaw
    Rot0   = RPYtoRot_ZXY(phi0, theta0, psi0)
    Quat0  = RotToQuat(Rot0)
    s[0]  = s_start.pos[0] #x
    s[1]  = s_start.pos[1] #y
    s[2]  = s_start.pos[2] #z
    s[3]  = s_start.vel[0] #xdot
    s[4]  = s_start.vel[1] #ydot
    s[5]  = s_start.vel[2] #zdot
    s[6]  = Quat0[0] #qw
    s[7]  = Quat0[1] #qx
    s[8]  = Quat0[2] #qy
    s[9] =  Quat0[3] #qz
    s[10] = 0        #p
    s[11] = 0        #q
    s[12] = 0        #r
    return s

def QuatToRot(q):
    """
    QuatToRot Converts a Quaternion to Rotation matrix written by Daniel Mellinger
    """
    # normalize q
    q = q / np.sqrt(np.sum(q**2))
    qahat = np.zeros([3, 3] )
    qahat[0, 1]  = -q[3]
    qahat[0, 2]  = q[2]
    qahat[1, 2]  = -q[1]
    qahat[1, 0]  = q[3]
    qahat[2, 0]  = -q[2]
    qahat[2, 1]  = q[1]
    R = np.identity(3) + 2 * qahat @ qahat + 2 * q[0] * qahat
    return R

def RotToQuat(R):
    """
    ROTTOQUAT Converts a Rotation matrix into a Quaternion written by Daniel Mellinger from the following website,
    deals with the case when tr<0 http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    """
    tr = np.sum(np.trace(R))
    if (tr > 0):
      S = np.sqrt(tr + 1.0) * 2 # S=4*qw
      qw = 0.25 * S
      qx = (R[2, 1] - R[1, 2]) / S
      qy = (R[0, 2] - R[2, 0]) / S
      qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
      S = np.sqrt(1.0 + R(1,1) - R(2,2) - R(3,3)) * 2 # S=4*qx
      qw = (R[2, 1] - R[1, 2]) / S
      qx = 0.25 * S
      qy = (R[0, 1] + R[1, 0]) / S
      qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1]  > R[2, 2] :
      S = np.sqrt(1.0 + R[1, 1]  - R[0, 0]  - R[2, 2] ) * 2 # S=4*qy
      qw = (R[0, 2]  - R[2, 0] ) / S
      qx = (R[0, 1]  + R[1, 0] ) / S
      qy = 0.25 * S
      qz = (R[1, 2]  + R[2, 1] ) / S
    else:
      S = np.sqrt(1.0 + R[2, 2]  - R[0, 0]  - R[1, 1] ) * 2 # S=4*qz
      qw = (R[1, 0]  - R[0, 1] ) / S
      qx = (R[0, 2]  + R[2, 0] ) / S
      qy = (R[1, 2]  + R[2, 1] ) / S
      qz = 0.25 * S
    q = np.array([[qw], [qx], [qy], [qz]])
    q = q * np.sign(qw)
    return q

def RPYtoRot_ZXY(phi, theta, psi):
    """
    RPYtoRot_ZXY Converts roll, pitch, yaw to a body-to-world Rotation matrix.
    The rotation matrix in this function is world to body [bRw] you will need to transpose this matrix to get the body
    to world [wRb] such that [wP] = [wRb] * [bP], where [bP] is a point in the body frame and [wP] is a point in the
    world frame written by Daniel Mellinger
    """
    R = np.array([[np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(psi) * np.sin(theta),
                   np.cos(theta)*np.sin(psi) + np.cos(psi)*np.sin(phi)*np.sin(theta), -np.cos(phi)*np.sin(theta)],
                  [-np.cos(phi)*np.sin(psi), np.cos(phi)*np.cos(psi), np.sin(phi)],
                  [np.cos(psi)*np.sin(theta) + np.cos(theta)*np.sin(phi)*np.sin(psi),
                   np.sin(psi)*np.sin(theta) - np.cos(psi)*np.cos(theta)*np.sin(phi), np.cos(phi)*np.cos(theta)]])

    return R

def RotToRPY_ZXY(R):
    """
    RotToRPY_ZXY Extract Roll, Pitch, Yaw from a world-to-body Rotation Matrix
    The rotation matrix in this function is world to body [bRw] you will need to transpose the matrix if you have a
    body to world [wRb] such that [wP] = [wRb] * [bP], where [bP] is a point in the body frame and [wP] is a point in
    the world frame written by Daniel Mellinger
    bRw = [ cos(psi)*cos(theta) - sin(phi)*sin(psi)*sin(theta),
            cos(theta)*sin(psi) + cos(psi)*sin(phi)*sin(theta),
            -cos(phi)*sin(theta)]
          [-cos(phi)*sin(psi), cos(phi)*cos(psi), sin(phi)]
          [ cos(psi)*sin(theta) + cos(theta)*sin(phi)*sin(psi),
             sin(psi)*sin(theta) - cos(psi)*cos(theta)*sin(phi),
               cos(phi)*cos(theta)]
    """
    phi = np.arcsin(R[1, 2])
    psi = np.arctan2(-R[1, 0] / np.cos(phi), R[1, 1] / np.cos(phi))
    theta = np.arctan2(-R[0, 2] / np.cos(phi), R[2, 2] / np.cos(phi))
    return phi, theta, psi

def qdToState(qd):
    """
     Converts state vector for simulation to qd struct used in hardware.
     x is 1 x 13 vector of state variables [pos vel quat omega]
     qd is a struct including the fields pos, vel, euler, and omega
    """
    x = np.zeros(13) #initialize dimensions
    x[0:3] = qd.pos
    x[3:6] = qd.vel
    Rot = RPYtoRot_ZXY(qd.euler[0], qd.euler[1], qd.euler[2])
    quat = RotToQuat(Rot)
    x[6:10] = quat
    x[11:13] = qd.omega
    return x

def stateToQd(x):
    """
    Converts qd struct used in hardware to x vector used in simulation
    x is 1 x 13 vector of state variables [pos vel quat omega]
    qd is a struct including the fields pos, vel, euler, and omega
    """
    qd = qd_object()
    # current state
    qd.pos = x[0:3]
    qd.vel = x[3:6]
    qd.Rot = QuatToRot(x[6:10])
    #print("Rot:\n", Rot)
    print("rotmat in my qd struct:\n", qd.Rot)
    [phi, theta, yaw] = RotToRPY_ZXY(qd.Rot)
    qd.euler = np.array([phi, theta, yaw])
    qd.omega = x[10:13]
    return qd


def diamond(t):
    """
    Desired diamond trajectory
    """
    T = 15

    if t < 0:
        pos = np.array([0, 0, 0])
        vel = np.array([0, 0, 0])
        acc = np.array([0, 0, 0])
    elif t < T / 4:
        pos = np.array([0, np.sqrt(2), np.sqrt(2)]) * t / (T / 4)
        vel = np.array([0, np.sqrt(2), np.sqrt(2)]) / (T / 4)
        acc = np.array([0, 0, 0])
    elif t < T / 2:
        pos = np.array([0, np.sqrt(2), np.sqrt(2)]) * (2 - 4 * t / T) + np.array([0, 0, 2 * np.sqrt(2)]) * (
                    4 * t / T - 1)
        vel = np.array([0, np.sqrt(2), np.sqrt(2)]) * (-4 / T) + np.array([0, 0, 2 * np.sqrt(2)]) * (4 / T)
        acc = np.array([0, 0, 0])
    elif t < 3 * T / 4:
        pos = np.array([0, 0, 2 * np.sqrt(2)]) * (3 - 4 * t / T) + np.array([0, -np.sqrt(2), np.sqrt(2)]) * (
                    4 * t / T - 2)
        vel = np.array([0, 0, 2 * np.sqrt(2)]) * (-4 / T) + np.array([0, -np.sqrt(2), np.sqrt(2)]) * (4 / T)
        acc = np.array([0, 0, 0])
    elif t < T:
        pos = np.array([0, -np.sqrt(2), np.sqrt(2)]) * (4 - 4 * t / T) + np.array([1, 0, 0.5]) * (4 * t / T - 3)
        vel = np.array([0, -np.sqrt(2), np.sqrt(2)]) * (-4 / T) + np.array([1, 0, 0]) * (4 / T)
        acc = np.array([0, 0, 0])
    else:
        pos = np.array([1, 0, 0.5])
        vel = np.array([0, 0, 0])
        acc = np.array([0, 0, 0])
    yaw = 0
    yawdot = 0

    desired_state = state_object()
    desired_state.pos = pos
    desired_state.vel = vel
    desired_state.acc = acc
    desired_state.yaw = yaw
    desired_state.yawdot = yawdot

    return desired_state

#############################################################

def vee_map(R):
    """
    Performs the vee mapping from a rotation matrix to a vector
    """
    arr_out = np.zeros(3)
    arr_out[0] = -R[1, 2]
    arr_out[1] = R[0, 2]
    arr_out[2] = -R[0, 1]
    return arr_out

def hat_map(a, mode = "torch"):
    if mode is "torch":
        a_hat = torch.tensor([[0, -a[2], a[1]],
                          [a[2], 0, -a[0]],
                          [-a[1], a[0], 0]], device=device, dtype=torch.float32)
    else:
        a_hat = np.array([[0, -a[2], a[1]],
                      [a[2], 0, -a[0]],
                      [-a[1], a[0], 0]])
    return a_hat


##############################################################


def plot_states1D(s_traj, s_plan, fig_num=None):
    """
    Plot position and velocity with each X, Y, Z dimension on a separate axis
    """

    plt.figure(fig_num, figsize=(10,7.5))
    ax_px = plt.subplot(421)
    ax_py = plt.subplot(423)
    ax_pz = plt.subplot(425)
    ax_yaw = plt.subplot(427)

    ax_vx = plt.subplot(422)
    ax_vy = plt.subplot(424)
    ax_vz = plt.subplot(426)
    ax_w =  plt.subplot(428)

    ax_px.plot(s_traj[:, -1], s_traj[:, 0])
    ax_px.plot(s_plan[:, -1], s_plan[:, 0])
    ax_px.set_ylabel('x (m)')

    ax_py.plot(s_traj[:, -1], s_traj[:, 1])
    ax_py.plot(s_plan[:, -1], s_plan[:, 1])
    ax_py.set_ylabel('y (m)')

    ax_pz.plot(s_traj[:, -1], s_traj[:, 2])
    ax_pz.plot(s_plan[:, -1], s_plan[:, 2])
    ax_pz.set_ylabel('z (m)')

    ax_vx.plot(s_traj[:, -1], s_traj[:, 3])
    ax_vx.plot(s_plan[:, -1], s_plan[:, 3])
    ax_vx.set_ylabel('x (m/s)')

    ax_vy.plot(s_traj[:, -1], s_traj[:, 4])
    ax_vy.plot(s_plan[:, -1], s_plan[:, 4])
    ax_vy.set_ylabel('y (m/s)')

    ax_vz.plot(s_traj[:, -1], s_traj[:, 5])
    ax_vz.plot(s_plan[:, -1], s_plan[:, 5])
    ax_vz.set_ylabel('z (m/s)')

    ax_yaw.plot(s_traj[:, -1], s_traj[:, 9])
    ax_yaw.plot(s_plan[:, -1], s_plan[:, 9])
    ax_yaw.set_ylabel('yaw (rad)')

    ax_w.plot(s_traj[:, -1], s_traj[:, 10])
    ax_w.plot(s_traj[:, -1], s_traj[:, 11])
    ax_w.plot(s_traj[:, -1], s_traj[:, 12])
    ax_w.plot(s_traj[:, -1], 0*s_traj[:, -1])
    ax_w.set_ylabel(r'$\omega$ (rad/s)')

    ax_px.set_title('Position/Yaw')
    ax_vx.set_title('Velocity')
    ax_yaw.set_xlabel('Time (s)')
    ax_w.set_xlabel('Time (s)')

    plt.subplots_adjust(left=0.1, right=0.98, top=0.93, wspace=0.3)
    #plt.savefig('./png/tracking_results.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()



def quadplot_update(s_traj, s_plan, t_curr=None):
    """
    Updates plot designated by an axis handle

    Note: s_traj will have np.nan values for any points not yet collected
    """
    if not(plt.isinteractive()):
        plt.ion()

    plt.figure()
    plt.clf()
    h_ax = plt.axes(projection='3d')
    h_ax.set_xlabel('x (m)')
    h_ax.set_ylabel('y (m)')
    h_ax.set_zlabel('z (m)')
    # Find min/max position values for each axis to normalize
    s_min = np.nanmin(s_traj[:, 0:3], axis=0)
    s_max = np.nanmax(s_traj[:, 0:3], axis=0)
    s_maxrange = np.max(s_max - s_min) * 1.1 # Add a 10% buffer to edges
    if s_maxrange < 2:
        s_maxrange = 2
    s_avg = (s_max + s_min) / 2

    # Plot valid points
    h_lines = h_ax.get_lines()
    if len(h_lines) < 2:
        h_ax.plot3D(s_traj[:, 0], s_traj[:, 1], s_traj[:, 2])
        h_ax.plot3D(s_plan[:, 0], s_plan[:, 1], s_plan[:, 2], '--')
    else:
        h_lines[0].set_data_3d(s_traj[:, 0], s_traj[:, 1], s_traj[:, 2])
        h_lines[1].set_data_3d(s_plan[:, 0], s_plan[:, 1], s_plan[:, 2])

    # Set equalized axis limits
    h_ax.set_xlim(s_avg[0] - s_maxrange / 2, s_avg[0] + s_maxrange / 2)
    h_ax.set_ylim(s_avg[1] - s_maxrange / 2, s_avg[1] + s_maxrange / 2)
    h_ax.set_zlim(s_avg[2] - s_maxrange / 2, s_avg[2] + s_maxrange / 2)

    if t_curr:
        h_ax.set_title('Simulation t = {0:2.3f}'.format(t_curr))
    h_ax.view_init(elev=25., azim=35)
    #plt.savefig('./png/traj.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show(block=True)


