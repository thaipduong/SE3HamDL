# Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control, RSS 2021
# Thai Duong, Nikolay Atanasov

# code structure follows the style of HNN by Greydanus et al. and SymODEM by Zhong et al.
# https://github.com/greydanus/hamiltonian-nn
# https://github.com/Physics-aware-AI/Symplectic-ODENet
import torch
from se3hamneuralode import MLP, PSD, MatrixNet


class SO3HamNODE(torch.nn.Module):
    '''
    Architecture for input (q, q_dot, u),
    where q represent quaternion, a tensor of size (bs, n),
    q and q_dot are tensors of size (bs, n), and
    u is a tensor of size (bs, 1).
    '''

    def __init__(self, M_net = None, V_net = None, g_net = None, device=None, u_dim=3, init_gain=1):
        super(SO3HamNODE, self).__init__()
        self.rotmatdim = 9
        self.angveldim = 3
        if M_net is None:
            self.M_net = PSD(self.rotmatdim, 300, self.angveldim, init_gain=init_gain).to(device)
        else:
            self.M_net = M_net
        if V_net is None:
            self.V_net = MLP(self.rotmatdim, 50, 1, init_gain=init_gain).to(device)
        else:
            self.V_net = V_net

        self.u_dim = u_dim
        if g_net is None:
            if u_dim == 1:
                self.g_net = MLP(self.rotmatdim, 200, self.angveldim).to(device)
            else:
                self.g_net = MatrixNet(self.rotmatdim, 300, self.angveldim * self.u_dim,
                                       shape=(self.angveldim, self.u_dim), init_gain=init_gain).to(device)
        else:
            self.g_net = g_net

        self.device = device
        self.nfe = 0

    def forward(self, t, x):
        with torch.enable_grad():
            self.nfe += 1
            bs = x.shape[0]
            zero_vec = torch.zeros(bs, self.u_dim, dtype=torch.float32, device=self.device)

            q, q_dot, u = torch.split(x, [self.rotmatdim, self.angveldim, self.u_dim], dim=1)
            M_q_inv = self.M_net(q)
            if self.rotmatdim == 1:
                p = q_dot / M_q_inv
            else:
                # assert 1==0
                q_dot_aug = torch.unsqueeze(q_dot, dim=2)
                p = torch.squeeze(torch.matmul(torch.inverse(M_q_inv), q_dot_aug), dim=2)

            q_p = torch.cat((q, p), dim=1)
            q, p = torch.split(q_p, [self.rotmatdim, self.angveldim], dim=1)
            M_q_inv = self.M_net(q)
            V_q = self.V_net(q)
            g_q = self.g_net(q)

            if self.rotmatdim == 1:
                H = p * p * M_q_inv / 2.0 + V_q
            else:
                p_aug = torch.unsqueeze(p, dim=2)
                H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug))) / 2.0 \
                    + torch.squeeze(V_q)
            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]
            dHdq, dHdp = torch.split(dH, [self.rotmatdim, self.angveldim], dim=1)

            if self.u_dim == 1:
                F = g_q * u
            else:
                F = torch.squeeze(torch.matmul(g_q, torch.unsqueeze(u, dim=2)))
            dq03 = torch.cross(q[:, 0:3], dHdp)
            dq36 = torch.cross(q[:, 3:6], dHdp)
            dq69 = torch.cross(q[:, 6:9], dHdp)
            dq = torch.cat((dq03, dq36, dq69), dim=1)
            dp = torch.cross(p, dHdp) + torch.cross(q[:, 0:3], dHdq[:, 0:3]) \
                                      + torch.cross(q[:, 3:6], dHdq[:, 3:6]) \
                                      + torch.cross(q[:, 6:9], dHdq[:, 6:9]) \
                                      + F

            dM_inv_dt = torch.zeros_like(M_q_inv)
            for row_ind in range(self.angveldim):
                for col_ind in range(self.angveldim):
                    dM_inv = torch.autograd.grad(M_q_inv[:, row_ind, col_ind].sum(), q, create_graph=True)[0]
                    dM_inv_dt[:, row_ind, col_ind] = (dM_inv * dq).sum(-1)
            ddq = torch.squeeze(torch.matmul(M_q_inv, torch.unsqueeze(dp, dim=2)), dim=2) \
                  + torch.squeeze(torch.matmul(dM_inv_dt, torch.unsqueeze(p, dim=2)), dim=2)

            return torch.cat((dq, ddq, zero_vec), dim=1)
