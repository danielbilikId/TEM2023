function HR_vec = continuous_HR(data,T_win,T_int,fs,method)
%The function generates a vector of vital signs estimates

heart_band = [40 90]./60;
L = length(data);
L_win = T_win*fs; % Number of data samples in selected window
t_win = linspace(0,L_win/fs,L_win);        % Time values in selected window
Iter_Num = floor((L-L_win)/(T_int*fs))+1; % Number of real-time computations
HR_vec = zeros(Iter_Num,1);

for i=1:Iter_Num 
    data_win = data(:,(i-1)*T_int*fs+1:L_win+(i-1)*T_int*fs);
    
    if method %1 is peak counting
       [pks,locs] = findpeaks(data_win,fs,'MinPeakHeight',0.1,'MinPeakDistance',0.67);
        HR_vec(i) = length(locs)*(60/T_win);
    else %0 is FFT based (not padded)
       [~,~,HR_vec(i)] = FFT_peak_selection(data_win,fs,[],heart_band,0);
    end
end

