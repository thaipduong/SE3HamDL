import numpy as np

from controller_utils import *
import torch, os, sys, argparse
from se3hamneuralode import from_pickle, SE3HamNODE

gpu=0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

class ControllerParams:
    def __init__(self):
        """
        Controller gains and set maximum tilt angle
        """
        self.maxangle = 40 * np.pi / 180  # you can specify the maximum commanded angle here
        self.K_p = 2*np.array([5, 5, 25])  # K_p
        self.K_v = 1.2*np.array([2.5, 2.5, 2.5])  # K_v
        self.K_R = 1 * np.array([250, 250, 250])  # K_R
        self.K_w = 1 * np.array([20, 20, 20])  # K_w

class LearnedEnergyBasedController:
    def __init__(self):
        self.model, self.stats = self.get_model()
        self.params = ControllerParams()

    def get_model(self):
        model = SE3HamNODE(device=device, pretrain=False).to(device)
        # path = 'data/trained_quadrotor_model.tar'
        path = 'data/quadrotor-se3ham-rk4-5p.tar'
        model.load_state_dict(torch.load(path, map_location=device))
        path = 'data/quadrotor-se3ham-rk4-5p-stats.pkl'
        stats = from_pickle(path)
        return model, stats



    def gen_control(self, qd, t):
        # Control gain
        K_p = self.params.K_p # K_p
        K_dv = self.params.K_v # K_v
        K_R = self.params.K_R # K_R
        K_dw = self.params.K_w # K_w

        # State
        pos = qd.pos  # y[0:3]
        R = qd.Rot
        RT = R.T
        # Convert velocity to body frame
        v = np.matmul(RT, qd.vel)
        w = qd.omega
        y = np.concatenate((pos, R.flatten()))
        pose = torch.tensor(y, requires_grad=True, dtype=torch.float64).to(device)
        pose= pose.view(1,12)
        x_tensor, R_tensor = torch.split(pose, [3, 9], dim=1)

        # Query the values of masses M1, M2, potential energy V, input coeff g.
        g_q = self.model.g_net(pose)
        V_q = self.model.V_net(pose)
        M1 = self.model.M_net1(x_tensor)
        M2 = self.model.M_net2(R_tensor)

        #
        g_pos = g_q.detach().cpu().numpy()[0]
        g_pos_T = g_pos.T
        g_pos_dagger = np.matmul(np.linalg.inv(np.matmul(g_pos_T, g_pos)), g_pos_T)
        M1_inv = M1.detach().cpu().numpy()[0]
        M1 = np.linalg.inv(M1_inv)
        M2_inv = M2.detach().cpu().numpy()[0]
        M2 = np.linalg.inv(M2_inv)
        dVdq = torch.autograd.grad(V_q, pose)[0]
        dVdq = dVdq.detach().cpu().numpy()[0]
        w_hat = hat_map(w, mode = "numpy")
        pv = np.matmul(M1, v)
        pw = np.matmul(M2, w)

        # Desired state
        pos_ref = qd.pos_des
        v_ref = qd.vel_des
        a_ref = qd.acc_des
        j_ref = np.zeros(3)
        yaw_ref = qd.yaw_des
        yaw_dot_ref = 0

        # Calculate thrust
        RTdV = np.matmul(RT,dVdq[0:3])
        pvxw = np.cross(pv, w)
        RTKp = np.matmul(M1, np.matmul(RT, K_p*(pos - pos_ref)))
        Kdvv_ref =  np.matmul(M1, K_dv*(v - np.matmul(RT, v_ref)))
        pvdot_ref = np.matmul(M1, np.matmul(RT, a_ref) - np.matmul(w_hat, np.matmul(RT, v_ref)))
        b_p_B = RTdV  - RTKp - Kdvv_ref + pvdot_ref - pvxw# + pvdot_ref # body frame
        b_p = np.matmul(R, b_p_B) # in world frame

        # Limit max tilt angle
        tiltangle = np.arccos(b_p[2] / np.linalg.norm(b_p))
        scale_acc = 1
        if tiltangle > self.params.maxangle:
            xy_mag = np.linalg.norm(b_p[:2])
            xy_mag_max = b_p[2] * np.tan(self.params.maxangle)
            scale_acc = xy_mag_max / xy_mag
            b_p[:2] = b_p[:2] * scale_acc
        b_p_B = np.matmul(RT, b_p)


        # # Calculate R_c
        # b1_ref = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0])
        # b3 = (b_p / np.linalg.norm(b_p))
        # b2 = np.cross(b3, b1_ref)
        # b2 = b2/np.linalg.norm(b2)
        # b1 = np.cross(b2, b3)
        # b1 = b1/np.linalg.norm(b1)
        # Rc = np.vstack((b1, b2, b3)).T
        #
        # # Calculate w_c
        # b_p_dot = scale_acc*np.matmul(M1, K_p*(v - np.matmul(RT, v_ref)) + np.matmul(RT, j_ref) )# + K_dv*(np.matmul(RT, a_ref)) \
        #           #+ np.matmul(M1, np.matmul(RT, j_ref) - np.matmul(w_hat, np.matmul(RT, a_ref)))
        # b_p_dot = np.matmul(R, b_p_dot) # back to the world frame
        # b_p_dot_norm = b_p_dot/np.linalg.norm(b_p)
        # b3_dot = np.cross(np.cross(b3, b_p_dot_norm), b3)
        # b1_ref_dot = np.array([-np.sin(yaw_ref) * yaw_dot_ref, np.cos(yaw_ref) * yaw_dot_ref, 0])
        # b2_dot = np.cross(np.cross(b2, (np.cross(b1_ref_dot, b3) + np.cross(b1_ref, b3_dot)) / np.linalg.norm(np.cross(b1_ref, b3))), b2)
        # b1_dot = np.cross(b3_dot, b2) + np.cross(b3, b2_dot)
        #
        # # Calculate R_c_dot
        # Rc_dot = np.vstack((b1_dot, b2_dot, b3_dot)).T
        # wc_hat = np.matmul(Rc.T, Rc_dot)
        # wc = np.array([wc_hat[2, 1], wc_hat[0, 2], wc_hat[1, 0]])
        # print("wc_hat", wc_hat)


        # Calculate R_c
        b2_ref = np.array([-np.sin(yaw_ref), np.cos(yaw_ref), 0])
        b3 = (b_p / np.linalg.norm(b_p))
        b1 = np.cross(b2_ref, b3)
        b1 = b1/np.linalg.norm(b1)
        b2 = np.cross(b3, b1)
        b2 = b2/np.linalg.norm(b2)
        Rc = np.vstack((b1, b2, b3)).T

        # Calculate w_c
        b_p_dot = scale_acc*np.matmul(M1, -K_p*(v - np.matmul(RT, v_ref)) + np.matmul(RT, j_ref) )# + K_dv*(np.matmul(RT, a_ref)) \
                  #+ np.matmul(M1, np.matmul(RT, j_ref) - np.matmul(w_hat, np.matmul(RT, a_ref)))
        b_p_dot = np.matmul(R, b_p_dot) # back to the world frame
        b_p_dot_norm = b_p_dot/np.linalg.norm(b_p)

        b3_dot = np.cross(np.cross(b3, b_p_dot_norm), b3)
        b2_ref_dot = np.array([np.cos(yaw_ref) * yaw_dot_ref, -np.sin(yaw_ref) * yaw_dot_ref, 0])
        b1_dot = np.cross(np.cross(b1, (np.cross(b2_ref_dot, b3) + np.cross(b2_ref, b3_dot)) / np.linalg.norm(np.cross(b2_ref, b3))), b1)
        b2_dot = np.cross(b3_dot, b1) + np.cross(b3, b1_dot)

        # Calculate R_c_dot
        Rc_dot = np.vstack((b1_dot, b2_dot, b3_dot)).T
        wc_hat = np.matmul(Rc.T, Rc_dot)
        wc = np.array([wc_hat[2, 1], wc_hat[0, 2], wc_hat[1, 0]])
        print("wc_hat", wc_hat)

        # Calculate b_R
        rxdV = np.cross(R[0,:], dVdq[3:6]) + np.cross(R[1,:], dVdq[6:9]) + np.cross(R[2,:], dVdq[9:12])
        pwxw = np.cross(pw, w)
        pvxv = np.cross(pv, v)
        e_euler = 0.5 * K_R* vee_map(Rc.T @ R - R.T @ Rc)
        kdwwc = K_dw*(w - np.matmul(RT, np.matmul(Rc, wc)))
        b_R = np.matmul(M2, - e_euler - kdwwc) - pwxw - pvxv + rxdV

        # Calculate the control
        wrench = np.hstack((b_p_B, b_R))
        control = np.matmul(g_pos_dagger, wrench)
        F = max(0.,control[0])
        M = control[1:]

        return F, M