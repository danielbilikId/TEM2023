from fri import VPW_FRI,FRI_estimator
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import iaf
import scipy.io
def format_1D(ax, title, label=''):
    ax.set_title(title, fontsize=25)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_ylabel(label, fontsize=25)


def plot_fri_noise(snr):
    gs = gridspec.GridSpec(1, 3)

    tk = np.array([0.6, 0.7, 0.3])
    rk = np.array([0.035, 0.02, 0.085])
    ck = np.array([0.75, 0.7, 5.0])
    T = 1.0
    K = 5

    # Standard VPW
    N = 2500

    vpw_fri = VPW_FRI(tk, rk, ck, T)
    time = np.linspace(0, T, N)
    frequencies = np.fft.fftfreq(int(N), T / N)
    omega = 2 * np.pi * frequencies / T
    signal = vpw_fri.evaluate_time_domain(time)
    spectrum = vpw_fri.evaluate_Fourier_domain(omega)


    signal = np.real(np.fft.ifft(spectrum))
    #mdic = {"a": signal}
    #scipy.io.savemat('vetterli.mat',mdic)
    signal_power = np.linalg.norm(signal) ** 2 / N
    noise_power = signal_power / np.power(10, snr / 10.0)
    u = scipy.io.loadmat('refrence.mat')
    u = u['x2']

    u = np.transpose(u)
    u = u[2500:2500 * 2]
    #signal = u
    fig = plt.figure(figsize=(18, 6), dpi=300)
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(hspace=0.01, wspace=0.05)

    dur = 1
    b = .2
    d = 0.07
    C = 50
    dt = 4e-4
    signal_noisy = signal + np.random.normal(0, np.sqrt(noise_power), int(N))
    s = iaf.iaf_encode(signal, dt, b=b, d=d,R= 1, C=C)
    yDel = -b * np.diff(s) + C * d
    Kmax = 4 * K + 2
    w0 = 2 * np.pi / dur
    Kmax_arr = np.arange(-Kmax, Kmax + 1)
    Kmax_arr = np.reshape(Kmax_arr, [1, len(Kmax_arr)])
    s = np.reshape(s, [len(s), 1])
    F = np.exp(1j * w0 * s[1:] * Kmax_arr) - np.exp(1j * w0 * s[0:-1] * Kmax_arr)
    F[:, K + 1] = np.transpose(s[1:] - s[0:- 1])
    ss = np.zeros((len(Kmax_arr), len(Kmax_arr)), dtype=np.complex)
    ss = dur / (1j * 2 * np.pi * Kmax_arr)
    ss[:, Kmax] = 1
    L = np.shape(Kmax_arr)[1]
    # ss = np.transpose(ss)
    a = np.zeros((L, L), dtype=np.complex)
    bb = np.zeros((L, L), dtype=np.complex)
    np.fill_diagonal(a, np.real(ss[:, 1]))
    np.fill_diagonal(bb, np.imag(ss[:, 1]))
    S = a + 1j * bb
    ytnHat = np.linalg.pinv(F.dot(S))
    #spectrum_noisy = np.conj(ytnHat[Kmax + 2:])
    #spectrum_noisy = np.transpose(spectrum_noisy)
    #print(len(s))
    #spectrum_noisy = np.fft.fft(signal_noisy)
    fri_estimated = FRI_estimator(5,T,T/N,T/N).estimate_parameters(spectrum)
    spectrum_estimated = fri_estimated.evaluate_Fourier_domain(omega)

    classical_vpw = plt.subplot(gs[0, 0])
    classical_vpw.plot(time, signal)


    recovered_vpw = plt.subplot(gs[0, 1])
    recovered_vpw.plot(time, signal_noisy)



    signal_estimated = np.real(np.fft.ifft(spectrum_estimated))
    mse = np.mean(np.power(signal - signal_estimated, 2.0))
    snr_recovered = 10 * np.log10(signal_power / (np.linalg.norm(signal - signal_estimated) ** 2 / N))
    # print(snr_recovered)

    recovered_vpw = plt.subplot(gs[0, 2])
    recovered_vpw.plot(time, signal_estimated)

    plt.show()
plot_fri_noise(snr=10)