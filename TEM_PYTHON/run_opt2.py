#!/usr/bin/env python

import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import scipy.io
# Set matplotlib backend so that plots can be generated without a
# display:
import matplotlib
# matplotlib.use('AGG')
import pandas as pd
import band_limited as bl
import plotting as pl
import iaf


# region MSE
def calculate_mse(u, dur, dt, bw, bias, thresh, kappa, R=np.inf):
    s = iaf.iaf_encode(u, dt, bias, thresh, R, kappa)
    u_rec = iaf.iaf_decode(s, dur, dt, bw, bias, thresh, R, kappa)
    mse = mean_squared_error(u, u_rec)

    return mse


def run_mse(params, u, dur, dt, bw):
    return calculate_mse(u, dur, dt, bw, params[0], params[1], params[2])


# endregion

# region Constraints functions
def recoverable(x, c, bw):
    return 1 - (x[1] * x[2] * bw) / ((x[0] - c) * np.pi)


def bias_bound(x, c):
    return x[0] - c


# endregion


if __name__ == '__main__':
    # For determining output plot file names:
    output_name = 'iaf_demo_best'
    output_ext = '.png'

    # Define input signal:
    dur = 1
    dt = 1/2000
    f = 30
    bw = 2 * np.pi * f
    t = np.arange(0, dur, dt)

    np.random.seed(0)

    noise_power = None
    u = bl.gen_band_limited(dur, dt, f, noise_power)
    c = np.max(np.abs(u))

    #u = scipy.io.loadmat('signal_123_fri.dat')
    #u = u['rawSignal']
    #u = np.transpose(u)
    #u = u[1500:2000]
    #u = scipy.io.loadmat('GDN0011_pulse1.mat')
    #u = u['dt_ecgl']
    #u = u.transpose()
    #plt.plot(u)
    #plt.show()
    u = scipy.io.loadmat('Patient_1_ecg_GDN0020.mat')
    u = u['x']
    u = u[0]
    n = 3
    # lazar parameters
    lazar_x = np.zeros(n)
    lazar_x[0] = 3.5
    lazar_x[1] = 0.7
    lazar_x[2] = 0.01

    # initial guess with lazar values
    # init_bias = lazar_x[0]
    # init_thresh = lazar_x[1]
    # init_kappa = lazar_x[2]

    # initial guess with thumb rules
    init_bias = 1.1 * c
    init_thresh = 0.5 * init_bias
    init_kappa = 0.01

    x0 = np.zeros(n)
    x0[0] = init_bias
    x0[1] = init_thresh
    x0[2] = init_kappa

    # show initial objective
    initial_mse = run_mse(x0, u, dur, dt, bw)
    print('Initial parameters: bias={:.3f}, thresh={:.3f}, kappa={:.3f}'.format(x0[0], x0[1], x0[2]))
    print('Initial MSE: ' + str(initial_mse))

    # parameters bounds
    bias_bnd = (c, 4)
    thresh_bnd = (1.0,1.1)
    kappa_bnd = (0.00001, 0.5)

    bnds = (bias_bnd, thresh_bnd, kappa_bnd)

    cons = [{'type': 'ineq', 'fun': recoverable, 'args': [c, bw]},
            {'type': 'ineq', 'fun': bias_bound, 'args': [c]}]

    # optimize
    solution = minimize(run_mse, x0, method='SLSQP',
                        bounds=bnds, constraints=cons, args=(u, dur, dt, bw),
                        options={'maxiter': 10000})
    x = solution.x

    # show final result
    final_mse = run_mse(x, u, dur, dt, bw)
    print('Final parameters: bias={:.3f}, thresh={:.3f}, kappa={:.3f}'.format(x[0], x[1], x[2]))
    print('Final MSE: ' + str(final_mse))
    improvment = 100 * (initial_mse - final_mse) / initial_mse
    print('Comparing to initial guess, MSE lowered by {:.2f}%'.format(improvment))

    lazar_mse = run_mse(lazar_x, u, dur, dt, bw)
    print('Lazar parameters: bias={:.3f}, thresh={:.3f}, kappa={:.3f}'.format(lazar_x[0], lazar_x[1], lazar_x[2]))
    print('Lazar MSE: ' + str(lazar_mse))
    improvment = 100 * (lazar_mse - final_mse) / lazar_mse
    print('Comparing to Lazar parameters, MSE lowered by {:.2f}%'.format(improvment))

    # plot result
    s = iaf.iaf_encode(u, dt, x[0], x[1], np.inf, x[2])
    u_rec = iaf.iaf_decode(s, dur, dt, bw, x[0], x[1], np.inf, x[2])
    fig_title = 'Optimal signal reconstruction with bias={:.3f}, thresh={:.3f}, kappa={:.3f}'.format(x[0], x[1], x[2])
    print(fig_title)
    pl.plot_compare(t, u, u_rec, fig_title,output_name + output_ext)
    plt.close()

    print('Done!')