function Differential_PREP(person)
ref = [];
est = [];
SNR = 16; 
RMSE_all = []; 
Files=dir('C:\Users\sampl_lab\Documents\MATLAB\HRM_ecg');
for kkkk=person+2
   RMSE =[]; 
   FileNames=Files(kkkk).name;
   disp(FileNames)
   signal = load("C:\Users\sampl_lab\Documents\MATLAB\HRM_ecg\"+FileNames);
   x = signal.tfm_ecg2';
    opol = 6;
    t = 1:length(x);
    [p,s,mu] = polyfit(t,x,opol);
    f_y = polyval(p,t,[],mu);
    x = x-f_y; 
 %x = diff(x); 
    for tt = 0:10000:length(x)-10000
        x(tt+1:tt+10000) = detrend(x(tt+1:tt+10000));
        opol = 6;
        t = 1:10000;
        [p,s,mu] = polyfit(t,x(tt+1:tt+10000),opol);
        f_y = polyval(p,t,[],mu);
        x(tt+1:tt+10000) = x(tt+1:tt+10000)-f_y; 
    end
    signal2 = zeros(1,length(x));
    for idx = 1:2000:1200000-2000
     K = 5; %K= 5
    T = 1; 
    Kmax = 3*K+2;
    x_sig = x(idx+1:idx+2000);
    x_sig = smoothdata(x_sig); 
    N = length(x_sig); 
    %x_sig = x_sig+(10.^(-SNR/10)*randn(1,length(x_sig)));
    x_sig = x_sig/max(x_sig); 
    t = 0:1/length(x_sig):1-1/length(x_sig);    
    T = 1; 
    K =7;
    x_sig = diff(x_sig); 
    x_sig = [0 x_sig]; 
    x_sig = x_sig/max(x_sig); 
    y = 0;
    T = 1.0007; 
    signal = x_sig; 
    signal1 = signal;
    signal_noisy = N*signal;%+.8*rand(1,length(signal));
    signal = signal_noisy/N; 
    f1 = 0:1:N/2-1; 
    f2 = -N/2:-1; 
    frequency = [f1 f2]; 
    omega = 2*pi*frequency/T;
    spectrum = 0; 
    spectrum = spectrum/T; 
    spectrum = fft(signal_noisy);
    signal = signal1;%real(ifft(spectrum));
    %signal = signal_noisy/N; 
    T1 = T/N; 
    T2 = T/N; 
    m  = 1:1:K*10; 
    spectrum  = spectrum(1:30);
    spectrum =conj(spectrum); 
    %spectrum = cadzow(spectrum,K-1,inf)';
    %esprit
    l = round(length(spectrum)/2)*2; 
    tr = flip(spectrum(1:(length(spectrum))/2+2));
    tc = spectrum((length(spectrum))/2+2:end);
    tt = toeplitz(tc,tr); 
     [U,S,V] = svd(tt);
     V = conj(V(:,1:K))';
     V(1,:) = -V(1,:); 
     m = length(tr); 
     V = V'; 
     v1= V(1:m-1,:); 
     v2 = V(2:m,:); 
     [v,w] = eig(pinv(v2)*v1);
     w = conj(w);
     ww = diag(w);
     uk = ww'; 
%%%%%
tk = T*atan2(imag(uk),real(uk))/(2*pi);
rk = -T*log(abs(uk))/(2*pi);
for k=1:K
    if rk(k)<=0 
        rk(k) = T/N; 
    end
end
ck = 1/T*pinv(vander2(uk,length(spectrum)))'*spectrum';
ck2 = fliplr(ck'); 
tk = mod(tk,T); 
tk2  =fliplr(tk); 
rk2 = fliplr(rk); 

t = 0:1/N:1-1/N;
 
ck = real(ck');
 
    signal = 0;
    for k = 1:1:K
        signal = signal+time_eval(ck,rk,tk,t,k);
    end
%signal = signal/max(signal); 
signal = cumtrapz(signal);
signal2(idx+1:idx+2000)= signal; 
%signal2(idx+1:idx+2000) = max(x_sig)*signal22/max(signal22);
  rmse = mean(sqrt(abs(x_sig-signal2(idx+1:idx+2000)).^2));
  RMSE = [RMSE rmse]; 
%plot(t,x_sig,'b',linewidth = 1); hold on; plot(t,(signal2(idx+1:idx+1200)),'--r',linewidth=1);grid on;legend('og','est');
    end 
% L= signal2>1;
% data2 = signal2; 
% data2(L) = 0; 
   est = [est;signal2/max(signal2)]; 
   ref = [ref;x/max(x)];
   RMSE_all = [RMSE_all;RMSE]; 
end
L= est>0.5; data2 = est; 
 data2(L) = 0;
% 
%  for tt = 1:30
%     data2(tt,:) =0.85*data2(tt,:)/max(data2(tt,:));
%  end 
save('est.mat','data2'); 
save('ref.mat','ref'); 
end






