import numpy as np
import pandas as pd
import scipy.io
from fri import VPW_FRI,FRI_estimator
import iaf
import scipy.io
np.random.seed(32)

u = scipy.io.loadmat('TEM_PYTHON\DSOX1204G_TEM_pat2.mat')
u = u['y1']


u = u+np.ones(np.shape(u))*np.min(u)
bias=1.75
thresh=1.133
kappa=0.014
bias=0.78
thresh=0.99
kappa=0.018


u[np.isnan(u)] = 0
u = scipy.signal.resample(u,50)
dt = 1/50
s = iaf.iaf_encode(u, dt, bias, thresh, np.inf, kappa)
ts = np.cumsum(s)
print(len(ts))
s= pd.DataFrame(ts)
print(len(s))
s.to_csv('signal.csv')
K = 7
T = 1.0
N = 50
frequencies = np.fft.fftfreq(int(N), T / N)
omega = 2 * np.pi * frequencies / T
spectrum_noisy = scipy.io.loadmat('TEM_PYTHON/spectrum.mat')
spectrum_noisy = spectrum_noisy['spectrum']
print(len(spectrum_noisy))

spectrum_noisy = spectrum_noisy.reshape(len(spectrum_noisy),)
print(len(spectrum_noisy))
print(np.size(spectrum_noisy))
print(np.size(u))
fri_estimated = FRI_estimator(K, T, T / N, T / N).estimate_parameters_iqml2(u,spectrum_noisy)
fri_estimated = fri_estimated[0]
#fri_estimated = FRI_estimator(K, T, T / N, T / N).estimate_parameters(spectrum_noisy)
spectrum_estimated = fri_estimated.evaluate_Fourier_domain(omega)
signal_estimated = np.real(np.fft.ifft(spectrum_estimated))
SRR = 20*np.log10(np.linalg.norm(u/np.max(u)-np.mean(u/np.max(u)))/np.linalg.norm(u/np.max(u)-signal_estimated/np.max(signal_estimated)))
RMSE = np.square(np.subtract(signal_estimated/np.max(signal_estimated),(u-np.mean(u))/np.max(u-np.mean(u)))).mean()
print(RMSE)
print(signal_estimated)
save_file= pd.DataFrame((signal_estimated)/np.max(signal_estimated))
save_file.to_csv('TEM_PYTHON/pulse2.csv')
