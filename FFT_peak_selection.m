function [time,RR,HBR] = FFT_peak_selection(signal,slow_time_rate,resp_freq,heart_freq, padding_factor)
tic

if padding_factor % if using a zero-padded signal
    freq_res = 1/60;
    freq_axis = 0:freq_res:slow_time_rate/2-freq_res;    % high resolution frequency axis
    pad_sig_length = padding_factor*length(signal); %length of padded signal 
    sig_fft = fft(signal,pad_sig_length);%sig_fft = pmusic(signal,1000,freq_axis,slow_time_rate);
else
    freq_axis = 0:slow_time_rate/length(signal):slow_time_rate/2-slow_time_rate/length(signal);      % Slow time frequency axis in time window
    sig_fft = fft(signal);
end

sig_fft = sig_fft(1:length(freq_axis)); %Usually length=L/2
mag_fft = abs(sig_fft);
% figure; plot(freq_axis*60,mag_fft);xlim(heart_freq.*60);

% peak_resp = max(mag((freq_axis >= resp_freq(1)) & (freq_axis <= resp_freq(2))));
% peak_resp_idx = find(mag == peak_resp,1);
% RR = freq_axis(peak_resp_idx)*60;
RR=0;

peak_heart = max(mag_fft(((freq_axis >= heart_freq(1)) & (freq_axis <= heart_freq(2)))));
peak_heart_idx = find(mag_fft == peak_heart,1);
HBR = freq_axis(peak_heart_idx)*60;

time = toc;
end

