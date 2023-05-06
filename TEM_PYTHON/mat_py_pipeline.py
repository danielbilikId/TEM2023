import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from fri import VPW_FRI,FRI_estimator
import iaf
import scipy.io
np.random.seed(32)
#u = scipy.io.loadmat('Patient_1_ecg_GDN0020.mat')
#u = u['x']
#u = u[0]
u = scipy.io.loadmat('TEM_PYTHON/DSOX1204G_TEM_pat1_90TN.mat')
u = u['y1']
#u = scipy.io.loadmat('1.mat')
#u = u['tfm_ecg2']
#u = u[690000:691200]

#u = scipy.io.loadmat('signal_123_fri.dat')
#u = u['rawSignal']
#u = np.transpose(u)
#u = u[1500:2000]
#u = np.transpose(u)
#u = u-np.mean(u)
#u = u[1118+509:1118+1018]
u = u+np.ones(np.shape(u))*np.min(u)
#u = u+np.random.random([1600,1])
#u = scipy.signal.resample(u,20000)
#u = scipy.io.loadmat('signal_123_fri.dat')
#u = u['rawSignal']
#u = u.transpose()
#u = u[1500:2000]
bias=1.75
thresh=1.133
kappa=0.014
bias=1.892
thresh=1.033
kappa=0.018
#u = scipy.signal.detrend(np.transpose(u),type='linear')
#u = u.transpose()
#u = signal

#dt = sampling interval
#u = scipy.io.loadmat('ecgHW6.mat')
#u = u['C']
u[np.isnan(u)] = 0
#u = u+10**(-50/10)*np.random.random([2000,1])
u = scipy.signal.resample(u,200)
dt = 1/200
s = iaf.iaf_encode(u, dt, bias, thresh, np.inf, kappa)
ts = np.cumsum(s)
print(len(ts))
s= pd.DataFrame(ts)
print(len(s))
s.to_csv('signal.csv')
K = 7
T = 1.0
N = 200
frequencies = np.fft.fftfreq(int(N), T / N)
omega = 2 * np.pi * frequencies / T
spectrum_noisy = scipy.io.loadmat('TEM_PYTHON\spectrum.mat')
spectrum_noisy = spectrum_noisy['spectrum']
print(type(spectrum_noisy))

spectrum_noisy = spectrum_noisy.reshape(len(spectrum_noisy),)
print(len(spectrum_noisy))
print(np.size(spectrum_noisy))
print(np.size(u))
fri_estimated = FRI_estimator(K, T, T / N, T / N).estimate_parameters_iqml2(u,spectrum_noisy)
fri_estimated = fri_estimated[0]
#fri_estimated = FRI_estimator(K, T, T / N, T / N).estimate_parameters(spectrum_noisy)
spectrum_estimated = fri_estimated.evaluate_Fourier_domain(omega)
#gs = gridspec.GridSpec(1, 2)
#classical_vpw = plt.subplot(gs[0, 0])
#time = np.linspace(0, T, N)
#classical_vpw.plot(time,u/np.max(u))
signal_estimated = np.real(np.fft.ifft(spectrum_estimated))
#signal_estimated = scipy.signal.savgol_filter(signal_estimated, 10, 3)
#max_loc = np.where(signal_estimated==signal_estimated.max())
#min_loc = np.where(signal_estimated==signal_estimated.min())
#temp = signal_estimated[max_loc]
#signal_estimated[max_loc] = np.abs(signal_estimated[min_loc])
#signal_estimated[min_loc] = -signal_estimated[max_loc]
#signal_estimated[signal_estimated<-0.5] = 0
#recovered_vpw = plt.subplot(gs[0, 1])
#recovered_vpw.plot(time,scipy.signal.detrend((signal_estimated-0.2)/np.max(signal_estimated-0.2))/np.max(scipy.signal.detrend((signal_estimated-0.2)/np.max(signal_estimated-0.2))))
#recovered_vpw.plot(time,(u)/np.max(u))
#recovered_vpw.plot(time,(signal_estimated)/np.max(signal_estimated))
#recovered_vpw.set_ylim(-0.5, 1.05 * np.max(signal_estimated))
#plt.show()
SRR = 20*np.log10(np.linalg.norm(u/np.max(u)-np.mean(u/np.max(u)))/np.linalg.norm(u/np.max(u)-signal_estimated/np.max(signal_estimated)))
RMSE = np.square(np.subtract(signal_estimated/np.max(signal_estimated),(u-np.mean(u))/np.max(u-np.mean(u)))).mean()
print(RMSE)
save_file= pd.DataFrame((signal_estimated-0.09)/np.max(signal_estimated-0.09))
save_file.to_csv('pulse1_hw_recon.csv')