from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime
from fri import VPW_FRI,FRI_estimator
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import iaf
import scipy.io
from fri import *
import random
import pandas as pd
def format_plot_noisy(title='', filename='../figures/untitled.pdf', average=False):
    plt.xlabel('Input SNR (dB)', ha='right', va='center', x=1, family='serif')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    if average:
        plt.ylabel(r"Average $\Delta t_k$", family='serif')
    else:
        plt.ylabel(r"$\Delta t_k$", family='serif')

    plt.title(title)

    plt.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.25)


def get_fisher_inf_matrix_inv(M, N, tk, rk, ck):
    m = np.arange(0, M)

    grad_y = np.zeros((3, N))

    grad_Y1 = -ck * 2j * np.pi * m * np.exp(-2 * np.pi * m * (rk + 1j * tk))
    grad_Y2 = -ck * 2 * np.pi * m * np.exp(-2 * np.pi * m * (rk + 1j * tk))
    grad_Y3 = np.exp(-2 * np.pi * m * (rk + 1j * tk))

    grad_y[0, :] = np.fft.irfft(grad_Y1, n=N)
    grad_y[1, :] = np.fft.irfft(grad_Y2, n=N)
    grad_y[2, :] = np.fft.irfft(grad_Y3, n=N)

    fim = np.zeros((3, 3))

    for n in np.arange(0, N):
        fim += np.outer(grad_y[:, n], grad_y[:, n])

    return np.linalg.inv(fim)


def get_mse_bias_tk(signal, number_repetitions, methods_names, snrs, K, T, N, M, tk, CRB=True):
    # Important to re-initialize the seed, otherwise all threads share the same state
    random.seed()

    signal_power = np.linalg.norm(signal) ** 2 / N

    if CRB:
        bias_tks = np.zeros((len(methods_names) + 1, len(snrs)))
        mse_tks = np.zeros((len(methods_names) + 1, len(snrs)))
    else:
        bias_tks = np.zeros((len(methods_names), len(snrs)))
        mse_tks = np.zeros((len(methods_names), len(snrs)))

    fri_est = FRI_estimator(K, T, T / N, T / N)
    G = fri_est.construct_G_iqml(signal)

    for n in range(number_repetitions):

        for j, snr in enumerate(snrs):
            noise_power = signal_power / np.power(10, snr / 10.0)

            signal_noisy = signal + np.random.normal(0, np.sqrt(noise_power), N)
            spectrum_noisy = np.fft.fft(signal_noisy)

            # print signal_noisy.shape
            # print spectrum_noisy[:number_coeffs].shape

            for i, method_name in enumerate(methods_names):
                # get the estimation method name
                method = getattr(FRI_estimator, 'estimate_parameters_' + method_name)
                if method_name == 'iqml':
                    fri_estimated = method(fri_est, signal_noisy, spectrum_noisy[:M], G=G)[0]
                else:
                    fri_estimated = method(fri_est, spectrum_noisy[:M])

                # sort the tks
                if len(fri_estimated.tk) > 1:
                    fri_estimated.tk = np.sort(fri_estimated.tk)

                bias_tks[i, j] += np.sum(tk - fri_estimated.tk)
                mse_tks[i, j] += np.sum(np.abs(tk - fri_estimated.tk) ** 2)

    return mse_tks / (number_repetitions * len(tk)), bias_tks / (number_repetitions * len(tk))


def cramer_rao_bound(number_repetitions, parallel=False):
    n_snrs = 40
    snrs = np.linspace(-10, 30, n_snrs)

    M = 50

    tk = np.array([0.15])
    tk = np.array([random.random()])
    rk = np.array([0.02])
    ck = np.array([0.7])
    T = 1.0
    K = len(tk)

    N = 50
    #N = 2 * M - 1

    methods_long = ['Pisarenko', 'Cadzow', 'Matrix-pencil', 'Pan', 'Cramer-Rao bound']
    methods_short = ['pisarenko', 'cadzow', 'esprit', 'iqml']

    bias_tks = np.zeros((len(methods_long), n_snrs))
    mse_tks = np.zeros((len(methods_long), n_snrs))

    fim_inv = get_fisher_inf_matrix_inv(M, N, tk, rk, ck)

    vpw_fri = VPW_FRI(tk, rk, ck, T)

    time = np.linspace(0, T - 1.0 / N, N)
    frequencies = np.fft.fftfreq(int(N), T / N)
    omega = 2 * np.pi * frequencies / T

    spectrum = vpw_fri.evaluate_Fourier_domain(omega)
    signal = np.real(np.fft.ifft(spectrum))
    # signal_power = np.linalg.norm(spectrum)**2/N
    signal_power = np.linalg.norm(signal) ** 2 / N

    # Our estimation (parallel implementation)
    if parallel:
        num_jobs = multiprocessing.cpu_count()
        num_it = int(np.ceil(number_repetitions / num_jobs))

        res = Parallel(n_jobs=num_jobs, verbose=0)(
            delayed(get_mse_bias_tk)(signal, num_it, methods_short, snrs, K, T, N, M, tk) for _ in range(num_jobs))
        mse, bias = zip(*res)

        for idx in range(num_jobs):
            mse_tks += mse[idx] / num_jobs
            bias_tks += bias[idx] / num_jobs

    # (non-parallel implementation)
    else:
        m, b = get_mse_bias_tk(signal, number_repetitions, methods_short, snrs, K, T, N, M, tk)
        mse_tks += m
        bias_tks += b

    # compute variances
    var_tks = mse_tks - bias_tks ** 2

    # CRB
    for i, snr in enumerate(snrs):
        noise_power = signal_power / np.power(10, snr / 10.0)
        fim_inv_snr = noise_power * fim_inv

        var_tks[-1, i] = fim_inv_snr[0, 0]
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    for idx, method_name in enumerate(methods_long[:-1]):
        plt.semilogy(snrs, np.sqrt(var_tks[idx, :]), label=method_name, color=plt.cm.Blues(1 - 0.2 * (idx + 1)))
        s = pd.DataFrame(np.sqrt(var_tks[idx, :]))
        s.to_csv(f"{idx}_1.csv")
    # Plot CRB
    plt.semilogy(snrs, np.sqrt(var_tks[-1, :]), '--', dashes=[2, 2], label=methods_long[-1], color='k')
    s = pd.DataFrame(np.sqrt(var_tks[-1, :]))
    s.to_csv('CRB_1.csv')
    # Format plot
    plt.legend(loc="upper right")
    format_plot_noisy(title='(a) Single VPW pulse')


    # MSE
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for idx, method_name in enumerate(methods_long[:-1]):
        plt.plot(snrs, bias_tks[idx, :], label=method_name, color=plt.cm.Blues(1 - 0.2 * (idx + 1)))
    # plt.plot(snrs, var_tks[-1,:], '--', dashes=[2,2], label=methods_long[-1], color='k')
    plt.legend(loc="upper right")
    format_plot_noisy(title='(a) Bias')

    # BIAS
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for idx, method_name in enumerate(methods_long[:-1]):
        plt.semilogy(snrs, np.sqrt(mse_tks[idx, :]), label=method_name, color=plt.cm.Blues(1 - 0.2 * (idx + 1)))
    plt.semilogy(snrs, np.sqrt(var_tks[-1, :]), '--', dashes=[2, 2], label=methods_long[-1], color='k')
    plt.legend(loc="upper right")
    format_plot_noisy(title='(a) MSE')
    plt.show()

start = datetime.now()

cramer_rao_bound(2000, parallel=True)#500 times at M=50 -> 46 mins

end = datetime.now()
print(end - start)