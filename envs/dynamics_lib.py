#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""
Created on Thu Aug 15 18:53:58 2019

Author: Zhichao Li at UCSD ERL
"""

import numpy as np


def ufas(t, gvec, ug_vec):
    """ governor dynamics unicyle-like (x,y,theta) fully actuated system (ufas)
    gvec_dot = ug_vec
    """
    dzdt = ug_vec
    return dzdt


def unicycle_cartesian(t, zvec, uvec):
    """ robot dynamics unicyle in cartesian system (x, y, theta)
    """
    v, omega = uvec
    theta = zvec[2]
    dzdt = [v * np.cos(theta), v * np.sin(theta), omega]
    return dzdt


def double_integrator(t, xvec, uvec):
    """
    Double Integrator (Second-order fully actuated system)
    Input:
            y: system states
            u: acceleration command
    Output:
            dxdt: system dynamics at current moment
    """
    # system dynamics

    dxdt = [xvec[2], xvec[3], uvec[0], uvec[1]]

    return dxdt


def test_func1(t, y, c):
    """ test function
    LTI example
    dy/dt = -c*y
    """

    dy = np.zeros(1)
    dy[0] = -c * y
    return dy


def test_func1_greeting(t, y, c, const_str):
    """ test function with greeting message
    # dy/dt = -c*y
    """
    dy = np.zeros(1)
    dy[0] = -c * y
    print(const_str)
    return dy


def test_func2(t, y, c):
    """ test function with time varying example
    # TV example dy/dt  = exp(-2t) - 2y
    """
    
    dy = np.zeros((len(y)))
    dy[0] = np.exp(-2 * t) - 2 * y[0]
    return dy


def integrator(t, y, u):
    """integator system dynamics
    """
    dxdt = u

    return dxdt


def test_pendulum(t, zvec, uvec):
    """
    Pendulum Dynamics
    Function parameters
    Input:
            zvec: system states  [z1, z2]
            uvec: external input [u]
    Output:
            dzdt: system dynamics at current moment


    Model Parameters:
        gamma: given scalar
    """
    # PYTHON IS ANNOYING FOR SIZE 1 ARRAY
    if type(uvec) is np.ndarray and len(uvec) == 1:
        u = uvec[0]
    else:
        u = uvec

    # constant parameters
    gamma = 1
    z1, z2 = zvec
    # system dynamics
    dzdt = np.array([z2, z1 ** 2 + gamma * u + np.cos(z1) * z2])

    return dzdt
