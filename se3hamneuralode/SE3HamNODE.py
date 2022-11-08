# Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control, RSS 2021
# Thai Duong, Nikolay Atanasov

# code structure follows the style of HNN by Greydanus et al. and SymODEM by Zhong et al.
# https://github.com/greydanus/hamiltonian-nn
# https://github.com/Physics-aware-AI/Symplectic-ODENet

import torch
import numpy as np

from se3hamneuralode import MLP, PSD, MatrixNet
from se3hamneuralode import compute_rotation_matrix_from_quaternion
from .utils import L2_loss



class SE3HamNODE(torch.nn.Module):
    def __init__(self, device=None, pretrain = True, M_net1  = None, M_net2 = None, V_net = None, g_net = None, udim = 4):
        super(SE3HamNODE, self).__init__()
        init_gain = 0.001
        self.xdim = 3
        self.Rdim = 9
        self.linveldim = 3
        self.angveldim = 3
        self.posedim = self.xdim + self.Rdim #3 for position + 12 for rotmat
        self.twistdim = self.linveldim + self.angveldim #3 for linear vel + 3 for ang vel
        self.udim = udim
        if M_net1 is None:
            self.M_net1 = PSD(self.xdim, 400, self.linveldim, init_gain=init_gain).to(device)
        else:
            self.M_net1 = M_net1
        if M_net2 is None:
            self.M_net2 = PSD(self.Rdim, 400, self.twistdim - self.linveldim, init_gain=init_gain).to(device)
        else:
            self.M_net2 = M_net2
        if V_net is None:
            self.V_net = MLP(self.posedim, 400, 1, init_gain=init_gain).to(device)
        else:
            self.V_net = V_net
        if g_net is None:
            self.g_net = MatrixNet(self.posedim, 400, self.twistdim*self.udim, shape=(self.twistdim,self.udim), init_gain=init_gain).to(device)
        else:
            self.g_net = g_net
        self.device = device
        self.nfe = 0
        if pretrain:
            self.pretrain()

    def pretrain(self):
        x = np.arange(-10, 10, 0.5)
        y = np.arange(-10, 10, 0.5)
        z = np.arange(-10, 10, 0.5)
        n_grid = len(z)
        batch = n_grid ** 3
        xx, yy, zz = np.meshgrid(x, y, z)
        Xgrid = np.zeros([batch, 3])
        Xgrid[:, 0] = np.reshape(xx, (batch,))
        Xgrid[:, 1] = np.reshape(yy, (batch,))
        Xgrid[:, 2] = np.reshape(zz, (batch,))
        Xgrid = torch.tensor(Xgrid, dtype=torch.float64).view(batch, 3).to(self.device)
        # Pretain M_net1
        m_net1_hat = self.M_net1(Xgrid)
        # Train M_net1 to output identity matrix
        m_guess = torch.eye(3)
        m_guess = m_guess.reshape((1, 3, 3))
        m_guess = m_guess.repeat(batch, 1, 1).to(self.device)
        optim1 = torch.optim.Adam(self.M_net1.parameters(), 1e-3, weight_decay=0.0)
        loss = L2_loss(m_net1_hat, m_guess)
        print("Start pretraining Mnet1!", loss.detach().cpu().numpy())
        step = 1
        while loss > 1e-6:
            loss.backward()
            optim1.step()
            optim1.zero_grad()
            if step%10 == 0:
                print("step", step, loss.detach().cpu().numpy())
            m_net1_hat = self.M_net1(Xgrid)
            loss = L2_loss(m_net1_hat, m_guess)
            step = step + 1
        print("Pretraining Mnet1 done!", loss.detach().cpu().numpy())
        # delete Xgrid to save memory
        del Xgrid
        torch.cuda.empty_cache()

        # Pretrain M_net2
        batch = 250000
        # Uniformly generate quaternion using http://planning.cs.uiuc.edu/node198.html
        rand_ =np.random.uniform(size=(batch, 3))
        u1, u2, u3 = rand_[:,0], rand_[:, 1], rand_[:, 2]
        quat = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                              np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)])
        q_tensor = torch.tensor(quat.transpose(), dtype=torch.float64).view(batch, 4).to(self.device)
        R_tensor = compute_rotation_matrix_from_quaternion(q_tensor)
        R_tensor = R_tensor.view(-1, 9)
        m_net2_hat = self.M_net2(R_tensor)
        # Train M_net2 to output identity matrix
        inertia_guess = torch.eye(3)
        inertia_guess = inertia_guess.reshape((1, 3, 3))
        inertia_guess = inertia_guess.repeat(batch, 1, 1).to(self.device)
        optim = torch.optim.Adam(self.M_net2.parameters(), 1e-3, weight_decay=0.0)
        loss = L2_loss(m_net2_hat, inertia_guess)
        print("Start pretraining Mnet2!", loss.detach().cpu().numpy())
        step = 1
        while loss > 1e-6:
            loss.backward()
            optim.step()
            optim.zero_grad()
            if step%10 == 0:
                print("step", step, loss.detach().cpu().numpy())
            m_net2_hat = self.M_net2(R_tensor)
            loss = L2_loss(m_net2_hat, inertia_guess)
            step = step + 1
        print("Pretraining Mnet2 done!", loss.detach().cpu().numpy())
        # Delete data and cache to save memory
        del q_tensor
        torch.cuda.empty_cache()

    def forward(self, t, input):
        with torch.enable_grad():
            self.nfe += 1
            q, q_dot, u = torch.split(input, [self.posedim, self.twistdim, self.udim], dim=1)
            x, R = torch.split(q, [self.xdim, self.Rdim], dim=1)
            q_dot_v, q_dot_w = torch.split(q_dot, [self.linveldim, self.angveldim], dim=1)

            M_q_inv1 = self.M_net1(x)
            M_q_inv2 = self.M_net2(R)
            q_dot_aug_v = torch.unsqueeze(q_dot_v, dim=2)
            q_dot_aug_w = torch.unsqueeze(q_dot_w, dim=2)
            pv = torch.squeeze(torch.matmul(torch.inverse(M_q_inv1), q_dot_aug_v), dim=2)
            pw = torch.squeeze(torch.matmul(torch.inverse(M_q_inv2), q_dot_aug_w), dim=2)
            q_p = torch.cat((q, pv, pw), dim=1)
            q, pv, pw = torch.split(q_p, [self.posedim, self.linveldim, self.angveldim], dim=1)
            x, R = torch.split(q, [self.xdim, self.Rdim], dim=1)

            # Neural networks' forward passes
            M_q_inv1 = self.M_net1(x)
            M_q_inv2 = self.M_net2(R)
            V_q = self.V_net(q)
            g_q = self.g_net(q)

            # Calculate the Hamiltonian
            p_aug_v = torch.unsqueeze(pv, dim=2)
            p_aug_w = torch.unsqueeze(pw, dim=2)
            H = torch.squeeze(torch.matmul(torch.transpose(p_aug_v, 1, 2), torch.matmul(M_q_inv1, p_aug_v))) / 2.0 + \
                torch.squeeze(torch.matmul(torch.transpose(p_aug_w, 1, 2), torch.matmul(M_q_inv2, p_aug_w))) / 2.0 + \
                torch.squeeze(V_q)

            # Calculate the partial derivative using autograd
            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]
            # Order: position (3), rotmat (9), lin vel (3) in body frame, ang vel (3) in body frame
            dHdx, dHdR, dHdpv, dHdpw = torch.split(dH, [self.xdim, self.Rdim, self.linveldim, self.angveldim], dim=1)

            # Calculate g*u
            F = torch.squeeze(torch.matmul(g_q, torch.unsqueeze(u, dim=2)))

            # Hamilton's equation on SE(3) manifold for (q,p)
            Rmat = R.view(-1, 3, 3)
            dx = torch.squeeze(torch.matmul(Rmat, torch.unsqueeze(dHdpv, dim=2)))
            dR03 = torch.cross(Rmat[:, 0, :], dHdpw)
            dR36 = torch.cross(Rmat[:, 1, :], dHdpw)
            dR69 = torch.cross(Rmat[:, 2, :], dHdpw)
            dR = torch.cat((dR03, dR36, dR69), dim=1)
            dpv = torch.cross(pv, dHdpw) \
                  - torch.squeeze(torch.matmul(torch.transpose(Rmat, 1, 2), torch.unsqueeze(dHdx, dim=2))) \
                  + F[:, 0:3]
            dpw = torch.cross(pw, dHdpw) \
                  + torch.cross(pv, dHdpv) \
                  + torch.cross(Rmat[:, 0, :], dHdR[:, 0:3]) \
                  + torch.cross(Rmat[:, 1, :], dHdR[:, 3:6]) \
                  + torch.cross(Rmat[:, 2, :], dHdR[:, 6:9]) \
                  + F[:,3:6]

            # Hamilton's equation on SE(3) manifold for twist xi
            dM_inv_dt1 = torch.zeros_like(M_q_inv1)
            for row_ind in range(self.linveldim):
                for col_ind in range(self.linveldim):
                    dM_inv1 = \
                        torch.autograd.grad(M_q_inv1[:, row_ind, col_ind].sum(), x, create_graph=True)[0]
                    dM_inv_dt1[:, row_ind, col_ind] = (dM_inv1 * dx).sum(-1)
            dv = torch.squeeze(torch.matmul(M_q_inv1, torch.unsqueeze(dpv, dim=2)), dim=2) \
                  + torch.squeeze(torch.matmul(dM_inv_dt1, torch.unsqueeze(pv, dim=2)), dim=2)

            dM_inv_dt2 = torch.zeros_like(M_q_inv2)
            for row_ind in range(self.angveldim):
                for col_ind in range(self.angveldim):
                    dM_inv2 = \
                        torch.autograd.grad(M_q_inv2[:, row_ind, col_ind].sum(), R, create_graph=True)[0]
                    dM_inv_dt2[:, row_ind, col_ind] = (dM_inv2 * dR).sum(-1)
            dw = torch.squeeze(torch.matmul(M_q_inv2, torch.unsqueeze(dpw, dim=2)), dim=2) \
                  + torch.squeeze(torch.matmul(dM_inv_dt2, torch.unsqueeze(pw, dim=2)), dim=2)

            batch_size = input.shape[0]
            zero_vec = torch.zeros(batch_size, self.udim, dtype=torch.float64, device=self.device)
            return torch.cat((dx, dR, dv, dw, zero_vec), dim=1)