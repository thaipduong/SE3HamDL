
# python built in package
import os
import sys
import numpy as np
from numpy.linalg import norm
import matplotlib as mplt
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# third-party
import trimesh



class LaserSimError(Exception):
    """ User Defined Exceptions for Laser Simulation.
    """

    def __init__(self, *args):
        if args:
            self.msg = args[0]
        else:
            self.msg = ''

    def __str__(self):
        if self.msg:
            return "LaserSimError exception: {0}".format(self.msg)
        else:
            return "LaserSimError exception"

class LaserSim:
    """ Class of LIDAR Simulaiton. 3D -> 2D
    """

    def __init__(self, min_hang=-np.pi, max_hang=np.pi,
                 min_vang=-np.pi/12, max_vang=np.pi/12,
                 hcount=180, vcount=3,
                 min_rng=0.05, max_rng=30.0, noise_sd=0.0):
        """ Init LIDAR Specs, and cooridinate tranformation constant matrix
        for all elevation and azimuth.
        """
        # self.angle_limits \
        #            = np.array([min_hang, max_hang, min_vang, max_vang])
        # self.angle_counts = np.array([hcount,vcount])
        azimuth = min_hang + \
            np.arange(hcount)*(max_hang - min_hang)/(hcount)
        elevation = min_vang + \
            np.arange(vcount)*(max_vang - min_vang)/(vcount)
        ce = np.cos(elevation[:, None])
        se = np.tile(np.sin(elevation), (hcount, 1)).T
        # spherical to cartensian
        # x = r cos(elevation) cos(azimuth);
        # y = r cos(elevation) sin(azimuth)
        # z = r sin(elevation)
        self._vcount = vcount
        self._hcount = hcount
        self._azimuth = azimuth
        self._elevation = elevation
        self._min_rng = min_rng
        self._max_rng = max_rng

        self._frame = np.stack(
            [ce * np.cos(azimuth), ce * np.sin(azimuth), se], axis=2)
        self.range_limits = np.array([min_rng, max_rng])
        self._noise_sd = noise_sd
        self.dist_vec = []  # dvec = np.array([dgF, drg, drF])

    def get_range_scan3D(self, R, p, mesh):
        """Get 3D scan points rho of all rays, given LIDAR pose(R, p) and
        enviroment (mesh). Intersections between rays and mesh environment
        are obtained using trimesh built-in methods.
        """
        # rho = np.zeros(np.prod(self.angle_counts))
        # R is (3,3) frame_aug = (hcount,vcount,3,1)
        # broadcasting rule is R @ last two dimension (3,1)
        # res is (hcount * vcount, 3) each row is a laser scan point
        ray_directions = (R @ self._frame[..., None]).reshape((-1, 3))
        ray_origins = np.tile(p, (ray_directions.shape[0], 1))
        # these may be used to obtain color information too!
        # using mesh sometimes can be 0-volumne error
        # lidar scan can penetrate some place causing lidar range err!!!
        locations, index_ray, _ \
            = mesh.ray.intersects_location(ray_origins=ray_origins,
                                           ray_directions=ray_directions,
                                           multiple_hits=False)
        rho = np.inf*np.ones(ray_directions.shape[0])
        rho[index_ray] = np.sqrt(
            np.sum((locations - ray_origins[index_ray, :])**2, axis=1))
        rho = rho.reshape((self._frame.shape[0], self._frame.shape[1]))
        rho = rho[::-1, :]  # reverse row index for lower left origin
        return rho

    def get_range_2D(self, R, p, mesh):
        """ Extract 2D plane (z=0) by extracting 3D data
        """
        p3D = np.hstack((p, 0))
        rho_3D = LaserSim.get_range_scan3D(self, R, p3D, mesh)
        rho_zero = rho_3D[int(self._vcount/2)]
        return rho_zero

    def get_lidar_endpts(self, meshbounds, rho, p, dim=2, show_fig=False, eps=0.25, fig = None, ax=None):
        """ Get lidar scan end points coordinates in 2D/3D
        """
        if dim == 2:
            idx_LMR = rho < self._max_rng  # index less than maximum range
            # polar to cartesian 2D
            x = np.cos(self._azimuth[idx_LMR]) * rho[idx_LMR] + p[0]
            y = np.sin(self._azimuth[idx_LMR]) * rho[idx_LMR] + p[1]

            lidar_endpts = np.vstack((x, y)).T  # Num_pts * dim
            xmin, ymin = np.min(lidar_endpts, axis=0)
            xmax, ymax = np.max(lidar_endpts, axis=0)
            xL, xH = meshbounds[:, 0]
            yL, yH = meshbounds[:, 1]
            lidar_endpts = lidar_endpts[lidar_endpts[:, 0] >= xL - eps]
            lidar_endpts = lidar_endpts[lidar_endpts[:, 0] <= xH + eps]
            lidar_endpts = lidar_endpts[lidar_endpts[:, 1] >= yL - eps]
            lidar_endpts = lidar_endpts[lidar_endpts[:, 1] <= yH + eps]

            if show_fig is True:
                if fig is None or ax is None:
                    fig, ax = plt.subplots()
                ax.plot(p[0], p[1], 'gs')
                ax.plot(x, y, 'r*')
                ax.set_aspect('equal')
                ax.grid()
                return lidar_endpts, fig, ax
            else:
                return lidar_endpts, None, None

        else:
            raise LaserSimError("Haven't Implemented 3D Yet!")