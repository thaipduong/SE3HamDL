#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runge Kutta 4th Order Fixed Step Length ODE Solver
"""
import numpy as np


class RK4Error(Exception):
    """ User Defined Exceptions.
    """

    def __init__(self, *args):
        if args:
            self.msg = args[0]
        else:
            self.msg = ""

    def __str__(self):
        if self.msg:
            return "Runge Kutta 4th Order ODE Solver Error exception: {0}".format(self.msg)
        else:
            return "Runge Kutta 4th Order ODE Solver Error exception"


# function value evaluation
def feval(funcName, *args):
    return np.array(funcName(*args))


def rk4_update(h, func, tn, yn, un, *extra_func_args):
    """
    RK4 takes one Runge-Kutta step.
    This function is used to solve the forced initial value ode of the form
    dy/dt = f(t, y, uvec),  with y(t0) = yn
    User need to supply current values of t, y, a stepsize h, external input
    vector, and  dynamics of ackerman drive to evaluate the derivative,
    this function can compute the fourth-order Runge Kutta estimate
    to the solution at time t+h.

    #  Parameters:
    INPUT:
        func,   function handle     RHS of ODE equation,
                                    specifically ackerman drive dynamics

        tn,     scalar              the current time.
        yn,     1D array            y value at time tn
        un,     1D array            external input at time tn
        h,      scalar              step size, scalar


    OUTPUT:
        yn1,    1D array            the 4_th order Runge-Kutta estimate
                                    y value at next time step
    """
    # if type(yn) != np.ndarray:
    #     print('yn is ', end='')
    #     print(yn)
    #     raise RK4AckError('yn must be 1D numpy array')
    # else:
    #     pass

    # Get four sample values of the derivative.

    # print('DEBUG RK4, tn = %.4f, yn = %s, h = %.4f, un = %.4f'\
    #     %( tn, flist1D(yn), h, un) )
    k1 = feval(func, tn, yn, un, *extra_func_args)
    k2 = feval(func, tn + h / 2.0, yn + k1 * h / 2.0, un, *extra_func_args)
    k3 = feval(func, tn + h / 2.0, yn + k2 * h / 2.0, un, *extra_func_args)
    k4 = feval(func, tn + h, yn + k3 * h, un, *extra_func_args)

    # Estimate y at time t_n+1 using weight sum of 4 key sample pts
    yn1 = yn + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return yn1


if __name__ == "__main__":
    # from my_ack_fbk_utils import FlatSystem
    # from test_output_tracking_fdk_lin_pendulum_finished import pendulum_dynamics
    import matplotlib.pyplot as plt

    from dynamics_lib import integrator, test_func1, test_func1_greeting

    #%%  LTI example dy/dt = 1
    print("Runge-Kutta Solver dy/dt = -c y")
    h = 0.05  # step size
    tmax = 4
    tvec = np.linspace(0, tmax, int(tmax / h + 1))
    yvec = []
    y0 = 1 * np.ones(1)
    t0 = 0
    # intial value
    t = t0
    y = y0
    yvec.append(y0)

    # using rk4 to solve this using loop
    jj = 0
    while tvec[jj] < tmax:
        #    print('Iter %d' %(jj+1))
        const_str = "hello"
        y_new = rk4_update(h, test_func1, tvec[jj], y, -1)
        yvec.append(y_new)
        t = tvec[jj]
        y = y_new
        jj = jj + 1

    yvec_true = np.exp(tvec)

    plt.plot(tvec, yvec_true, "r", label="Exact solution")
    plt.plot(tvec, yvec, "b*", label="4th Order RK")
    plt.legend(loc="best")
    plt.title("Runge-Kutta Solver dy/dt = -c y", fontsize=12)
    # plt.text(x, y, s, fontsize=12)
    plt.xlabel("time")
    plt.tight_layout()
    plt.grid()
    plt.show()
    #%%  LTI example

    print("Runge-Kutta Solver test integrator, ut = cos(t)")
    h = 0.05  # step size
    tmax = 10
    tvec = np.linspace(0, tmax, int(tmax / h + 1))
    yvec = []
    y0 = np.zeros(1)
    t0 = 0
    # intial value
    t = t0
    y = y0
    c = 1  # u = 1 dxdt = u = 1
    yvec.append(y0)

    # using rk4 to solve this using loop
    jj = 0
    while tvec[jj] < tmax:
        #    print('Iter %d' %(jj+1))
        ut = np.cos(t)
        y_new = rk4_update(h, integrator, tvec[jj], y, ut)
        yvec.append(y_new)
        t = tvec[jj]
        y = y_new
        jj = jj + 1

    yvec_true = np.sin(tvec) - t0

    plt.plot(tvec, yvec_true, "r", label="Exact solution")
    plt.plot(tvec, yvec, "b", label="4th Order RK")
    plt.title("Runge-Kutta Solver test integrator, ut = cos(t)", fontsize=12)
    plt.legend(loc="best")
    plt.xlabel("time")
    plt.tight_layout()
    plt.grid()
    plt.show()
    #
    #
    ##%%  Ackerman Vehicle  2009 ECC dynmaic model
    #    print('Runge-Kutta Solver test Ackerman drive, constant input')
    #    h = 1/40.0 # step size for solver
    #    tmax = 20
    #    tvec = np.arange(0, tmax+h, h)
    #    zvec_log = []
    #    zvec0 = np.array([0, 0, 0, 0, 0])
    #
    #    z = zvec0 # intial value
    ##    print('Change u_ack to control the vehicle')
    #    u_ack = np.array([0.5 ,0.1])
    #    zvec_log.append(zvec0)
    #
    #    # using rk4 to solve this using loop
    #    tidx = 0
    ##    model_paras = [2.2, 2.2, 0.44]
    #    while tvec[tidx] < tmax:
    #    #    print('Iter %d' %(tidx+1))
    ##        z_new = rk4_update('FlatSystem.akerman_dynamic_2009ECC', tvec[tidx], z, h, u_ack)
    #
    #        M1 = 2.2  # kg
    #        M2 = 2.2  # kg
    #        L = 0.44  # m
    #
    #        z_new = rk4_update(h, akerman_dynamic_2009ECC, tvec[tidx], z, u_ack, M1, M2, L)
    #        zvec_log.append(z_new)
    #        t = tvec[tidx]
    #        z = z_new
    #        tidx = tidx + 1
    #
    #    sol = np.array(zvec_log)
    #
    #    fig1 = plt.figure()
    #    #plt.plot(t, sol[:, 0], 'b', label='x1(t)')
    #    #plt.plot(t, sol[:, 1], 'g', label='x2(t)')
    #    label_list = [r'$x_1$',r'$x_2$', r'$\theta$', r'$\phi$', r'$v_r$', r'$w$']
    #    for ii in range(len(zvec0)):
    #        plt.plot(tvec, sol[:, ii], label= label_list[ii])
    #
    #    plt.legend(loc='best')
    #    plt.xlabel('Time (seconds)')
    #    plt.grid()
    #    plt.show()
    #
    #    fig2 = plt.figure()
    #    plt.plot(sol[:, 0], sol[:, 1], 'g')
    #    plt.title('Position of the vehicle')
    #    plt.xlabel(r'$x_1(t)$')
    #    plt.ylabel(r'$x_2(t)$')
    #
    #    plt.grid()
    #    plt.show()
    #
    #%% Ackerman vehicle 6 states 4 + 2 [x, y, psi, delta_f, vr, ar]
    # get input from simulation
    #
    #    """ Read and anlyaze log result
    #    """
    #    filename1 = 'rss_cplx_c2_4.pkl'
    #    from gov_vis_rss_ver02 import GovLogViewer
    #    viewer1 = GovLogViewer(filename1)
    #    uack_log = viewer1.uack_log
    #    print('control history got succesfully!')
    #%%
    # print("Runge-Kutta Solver test Ackerman drive, constant input")
    # h = 1.0 / 40  # step size for solver
    # tmax = 0.5
    # tvec = np.arange(0, tmax + h, h)
    # zvec_log = []
    # zvec0 = np.array([1.0, 0.75, 0, 0, 0.1, 0])

    # z = zvec0  # intial value
    # #    print('Change u_ack to control the vehicle')
    # u_ack = np.array([0.5, 0.1])
    # zvec_log.append(zvec0)

    # # using rk4 to solve this using loop
    # tidx = 0
    # #    model_paras = [2.2, 2.2, 0.44]
    # while tvec[tidx] < tmax:
    #     #    print('Iter %d' %(tidx+1))
    #     #        z_new = rk4_update('FlatSystem.akerman_dynamic_2009ECC', tvec[tidx], z, h, u_ack)
    #     L = 0.44  # m
    #     #        uack = uack_log[tidx]
    #     u0 = np.sin(tidx * 0.1)
    #     u1 = np.cos(tidx * 0.1)

    #     uack = np.array([u0, u1])
    #     print("sack old %s" % flist1D(z))
    #     z_new = rk4_update(h, aug_akerman_dynamic, tvec[tidx], z, uack, L)
    #     print("sack new %s" % flist1D(z_new))
    #     zvec_log.append(z_new)
    #     t = tvec[tidx]
    #     z = z_new
    #     tidx = tidx + 1

    # sol = np.array(zvec_log)

    # fig1 = plt.figure()
    # # plt.plot(t, sol[:, 0], 'b', label='x1(t)')
    # # plt.plot(t, sol[:, 1], 'g', label='x2(t)')
    # #    label_list = [r'$x_1$',r'$x_2$', r'$\theta$', r'$\phi$', r'$v_r$', r'$w$']
    # #    for ii in range(len(zvec0)):
    # #        plt.plot(tvec, sol[:, ii], label= label_list[ii])

    # plt.legend(loc="best")
    # plt.xlabel("Time (seconds)")
    # plt.grid()
    # plt.show()

    # fig2 = plt.figure()
    # plt.plot(sol[:, 0], sol[:, 1], "g")
    # plt.title("Position of the vehicle")
    # plt.xlabel(r"$x_1(t)$")
    # plt.ylabel(r"$x_2(t)$")

    # plt.grid()
    # plt.show()
