# This is modified from: https://github.com/d-biswa/Symplectic-ODENet/blob/master/utils.py
# and https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py

import torch, pickle
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

################################ Utils ################################

def L2_loss(u, v):
    return (u-v).pow(2).mean()

def normalize_vector(v, return_mag=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if (return_mag == True):
        return v, v_mag[:, 0]
    else:
        return v

def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3
    return out

# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)

    theta = torch.acos(cos)
    return theta


def compute_rotation_matrix_from_unnormalized_rotmat(unnormalized_rotmat):
    x_raw = unnormalized_rotmat[:, 0:3]  # batch*3
    y_raw = unnormalized_rotmat[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 1, 3)
    y = y.view(-1, 1, 3)
    z = z.view(-1, 1, 3)
    matrix = torch.cat((x, y, z), 1)  # batch*3*3
    return matrix

def compute_geodesic_loss(gt_r_matrix, out_r_matrix):
    theta = compute_geodesic_distance_from_two_matrices(gt_r_matrix, out_r_matrix)
    theta = theta**2
    error = theta.mean()
    return error, theta


def compute_rotation_matrix_from_quaternion(quaternion):
    batch = quaternion.shape[0]

    quat = torch.nn.functional.normalize(quaternion)  # normalize_vector(quaternion).contiguous()

    qw = quat[..., 0].contiguous().view(batch, 1)
    qx = quat[..., 1].contiguous().view(batch, 1)
    qy = quat[..., 2].contiguous().view(batch, 1)
    qz = quat[..., 3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix
################################ Loss calculation for SO(3) ################################

def rotmat_L2_geodesic_loss(u,u_hat, split):
    q_hat, q_dot_hat, u_hat = torch.split(u_hat, split, dim=2)
    q, q_dot, u = torch.split(u, split, dim=2)
    qdot_u_hat = torch.cat((q_dot_hat, u_hat), dim=2).flatten(start_dim=0, end_dim=1)
    qdot_u = torch.cat((q_dot, u), dim=2).flatten(start_dim=0, end_dim=1)
    l2_loss = L2_loss(qdot_u, qdot_u_hat)
    q_hat = q_hat.flatten(start_dim=0, end_dim=1)
    q = q.flatten(start_dim=0, end_dim=1)
    R_hat = compute_rotation_matrix_from_unnormalized_rotmat(q_hat)
    R = compute_rotation_matrix_from_unnormalized_rotmat(q)
    geo_loss, _ = compute_geodesic_loss(R, R_hat)
    return l2_loss + geo_loss, l2_loss, geo_loss

def rotmat_L2_geodesic_diff(u,u_hat, split):
    q_hat, q_dot_hat, u_hat = torch.split(u_hat, split, dim=1)
    q, q_dot, u = torch.split(u, split, dim=1)
    qdot_u_hat = torch.cat((q_dot_hat, u_hat), dim=1)
    qdot_u = torch.cat((q_dot, u), dim=1)
    l2_diff = torch.sum((qdot_u - qdot_u_hat)**2, dim=1)
    R_hat = compute_rotation_matrix_from_unnormalized_rotmat(q_hat)
    R = compute_rotation_matrix_from_unnormalized_rotmat(q)
    _, geo_diff = compute_geodesic_loss(R, R_hat)
    return l2_diff + geo_diff, l2_diff, geo_diff

def traj_rotmat_L2_geodesic_loss(traj,traj_hat, split):
    total_loss = None
    l2_loss = None
    geo_loss = None
    for t in range(traj.shape[0]):
        u = traj[t,:,:]
        u_hat = traj_hat[t,:,:]
        if total_loss is None:
            total_loss, l2_loss, geo_loss = rotmat_L2_geodesic_diff(u,u_hat, split=split)
            total_loss = torch.unsqueeze(total_loss, dim=0)
            l2_loss = torch.unsqueeze(l2_loss, dim=0)
            geo_loss = torch.unsqueeze(geo_loss, dim=0)
        else:
            t_total_loss, t_l2_loss, t_geo_loss = rotmat_L2_geodesic_diff(u, u_hat, split=split)
            t_total_loss = torch.unsqueeze(t_total_loss, dim=0)
            t_l2_loss = torch.unsqueeze(t_l2_loss, dim=0)
            t_geo_loss = torch.unsqueeze(t_geo_loss, dim=0)
            total_loss = torch.cat((total_loss, t_total_loss), dim=0)
            l2_loss = torch.cat((l2_loss, t_l2_loss), dim=0)
            geo_loss = torch.cat((geo_loss, t_geo_loss), dim=0)
    return total_loss, l2_loss, geo_loss


################################ Loss calculation for SE(3) ################################

def pose_L2_geodesic_loss(u,u_hat, split):
    x_hat, R_hat, q_dot_hat, u_hat = torch.split(u_hat, split, dim=1)
    x, R, q_dot, u = torch.split(u, split, dim=1)
    x_qdot_u_hat = torch.cat((x_hat, q_dot_hat, u_hat), dim=1)
    x_qdot_u = torch.cat((x, q_dot, u), dim=1)
    l2_loss = L2_loss(x_qdot_u, x_qdot_u_hat)
    norm_R_hat = compute_rotation_matrix_from_unnormalized_rotmat(R_hat)
    norm_R = compute_rotation_matrix_from_unnormalized_rotmat(R)
    geo_loss, _ = compute_geodesic_loss(norm_R, norm_R_hat)
    return l2_loss + geo_loss, l2_loss, geo_loss

def pose_L2_geodesic_diff(u,u_hat, split):
    x_hat, R_hat, q_dot_hat, u_hat = torch.split(u_hat, split, dim=1)
    x, R, q_dot, u = torch.split(u, split, dim=1)
    x_qdot_u_hat = torch.cat((x_hat, q_dot_hat, u_hat), dim=1)
    x_qdot_u = torch.cat((x, q_dot, u), dim=1)
    l2_diff= torch.sum((x_qdot_u - x_qdot_u_hat)**2, dim=1)
    norm_R_hat = compute_rotation_matrix_from_unnormalized_rotmat(R_hat)
    norm_R = compute_rotation_matrix_from_unnormalized_rotmat(R)
    _, geo_diff = compute_geodesic_loss(norm_R, norm_R_hat)
    return l2_diff + geo_diff, l2_diff, geo_diff

def traj_pose_L2_geodesic_loss(traj,traj_hat, split):
    total_loss = None
    l2_loss = None
    geo_loss = None
    for t in range(traj.shape[0]):
        u = traj[t,:,:]
        u_hat = traj_hat[t,:,:]
        if total_loss is None:
            total_loss, l2_loss, geo_loss = pose_L2_geodesic_diff(u,u_hat, split=split)
            total_loss = torch.unsqueeze(total_loss, dim=0)
            l2_loss = torch.unsqueeze(l2_loss, dim=0)
            geo_loss = torch.unsqueeze(geo_loss, dim=0)
        else:
            t_total_loss, t_l2_loss, t_geo_loss = pose_L2_geodesic_diff(u, u_hat, split=split)
            t_total_loss = torch.unsqueeze(t_total_loss, dim=0)
            t_l2_loss = torch.unsqueeze(t_l2_loss, dim=0)
            t_geo_loss = torch.unsqueeze(t_geo_loss, dim=0)
            total_loss = torch.cat((total_loss, t_total_loss), dim=0)
            l2_loss = torch.cat((l2_loss, t_l2_loss), dim=0)
            geo_loss = torch.cat((geo_loss, t_geo_loss), dim=0)
    return total_loss, l2_loss, geo_loss




def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing


def choose_nonlinearity(name):
    nl = None
    if name == 'tanh':
        nl = torch.tanh
    elif name == 'relu':
        nl = torch.relu
    elif name == 'sigmoid':
        nl = torch.sigmoid
    elif name == 'softplus':
        nl = torch.nn.functional.softplus
    elif name == 'selu':
        nl = torch.nn.functional.selu
    elif name == 'elu':
        nl = torch.nn.functional.elu
    elif name == 'swish':
        nl = lambda x: x * torch.sigmoid(x)
    else:
        raise ValueError("nonlinearity not recognized")
    return nl
