function final_measurements = run_HRM()
f_ADC= 2000; 
T_win = 40;T_int =.5;%0.5
fs = f_ADC;
time_vec = T_win:T_int:duration; 
x= load('ref.mat');     
x = x.ref; 
signal2 = load('est.mat');
signal2 = signal2.data2; 
pref = []; 
final_SNR = []; 
for tt = 1
t = 1:length(x); 
method =0; %FFT
% method = 1; %peak counting

data = x(tt,:); 
HR_vec_ECG = continuous_HR(real(data),T_win,T_int,fs,method);

data = signal2(tt,:);
HR_vec_TEM = continuous_HR(real(data),T_win,T_int,fs,method);

HR_vec_TEM(237) = HR_vec_TEM(236); 
[performance(1,1),performance(1,2),performance(1,3),performance(1,4),performance(1,5)] = GT_comparison_TEM('HR','IF-TEM',real(HR_vec_ECG),real(HR_vec_TEM),time_vec,T_win,T_int);
set(gca,'FontSize',15)
disp(performance)
pref = [pref;performance];
end
final_measurements = median(pref,1);
disp(final_measurements);
end
