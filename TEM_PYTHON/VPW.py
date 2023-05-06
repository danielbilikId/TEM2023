from fri import *
from matplotlib import gridspec
import matplotlib.pyplot as plt
import scipy
import pandas as pd
snr = np.inf
gs = gridspec.GridSpec(1, 3)
T = 1.0
K = 7# Standard VPW
u = scipy.io.loadmat('ecg_best_ever_HW.mat')
u = u['C']
u = scipy.io.loadmat('signal_123_fri.dat')
u = u['rawSignal']
u = u.transpose()
u = u[1500:2000]
#u = scipy.io.loadmat('GDN0012_pulse2.mat')
#u = u['dt_ecgl']
u = scipy.io.loadmat('GDN0020_pulse3.mat')
u = u['dt_ecgl']
u = u.transpose()
#u = scipy.signal.resample(u,500)
u = scipy.signal.resample(u,500)
N = len(u)
#vpw_fri = u
u[np.isnan(u)] = 0
time = np.linspace(0, T, N)
frequencies = np.fft.fftfreq(int(N), T / N)
omega = 2 * np.pi * frequencies / T
signal = u
print(np.shape(signal))
signal = u
signal = np.reshape(u,[len(u),])
#signal = np.insert(signal,0,signal[0])
#signal = np.diff(signal)
print(np.shape(signal))
spectrum = np.fft.fft(signal)
fig = plt.figure(figsize=(18, 6), dpi=300)
fig.patch.set_facecolor('white')
plt.subplots_adjust(hspace=0.01, wspace=0.05)
signal_power = np.linalg.norm(signal) ** 2 / N
noise_power = signal_power / np.power(10, snr / 10.0)

signal_noisy = signal+ np.random.normal(0,np.sqrt(noise_power),int(N))
signal_noisy = np.insert(signal_noisy,0,signal_noisy[0])
signal_noisy = np.diff(signal_noisy)
spectrum_noisy = np.fft.fft(signal_noisy)
fri_estimated = FRI_estimator(K, T, T / N, T / N).estimate_parameters_iqml2(signal,spectrum_noisy[:30])
fri_estimated = fri_estimated[0]
#fri_estimated = FRI_estimator(K, T, T/N, T/N).estimate_parameters(spectrum_noisy[:30])
spectrum_estimated = fri_estimated.evaluate_Fourier_domain(omega)

classical_vpw = plt.subplot(gs[0, 0])
classical_vpw.plot(time, signal/np.max(signal))

recovered_vpw = plt.subplot(gs[0, 1])
recovered_vpw.plot(time, signal_noisy/np.max(signal_noisy))

signal_estimated = np.real(np.fft.ifft(spectrum_estimated))
signal_estimated = np.cumsum(signal_estimated)
mse = np.mean(np.power(signal - signal_estimated, 2.0))
snr_recovered = 10 * np.log10(signal_power / (np.linalg.norm(signal - signal_estimated) ** 2 / N))
# print(snr_recovered)

recovered_vpw = plt.subplot(gs[0, 2])
recovered_vpw.plot(time, signal_estimated/np.max(signal_estimated))
SRR = 20*np.log10(np.linalg.norm(u/np.max(u)-np.mean(u/np.max(u)))/np.linalg.norm(u/np.max(u)-signal_estimated/np.max(signal_estimated)))
print(SRR)
plt.show()

save_file= pd.DataFrame(signal_estimated)
#save_file.to_csv('signal_best_recon_diff_5.csv')