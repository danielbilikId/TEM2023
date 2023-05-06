function TEM_HRM_PREP(person)
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
    fs = 2000;
    x = x(1:1200000);%10 min of data
    %x = x/max(x); 
    dt_ecgl = detrend(x(1:10000));
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
    for idx = 1:2000:length(x)-2000
     K = 5; %K= 5
    T = 1; 

    Kmax = 3*K+2;

    x_sig = x(idx+1:idx+2000);
    %ac=max(x_sig);
    %peak=find(x_sig==ac); 
    %T =1.5*peak/2000; %works - somehow
    %x_sig(1:70) = 0; 
    x_sig = x_sig+10.^(-SNR/10)*randn(1,length(x_sig));
    x_sig = x_sig/max(x_sig); 
    y = 0;
    m  = -Kmax:1:Kmax; 
    G = zeros(1,length(m));
    N = length(x_sig);
    t = 0:1/N:1-1/N;
dt = 1/N; 
tLPF = -10:dt:10;
Kmax = 3*K+2;
    for n = 1:N
        G = G+x_sig(n)*exp(-2*pi/T*1i*n.*m./N); 
    end
    %real fourier coefficients positive side - G(K+1:end); 
    %G starts from 0; 

    %G = G/10e4;
    for i = 1:1:length(m)
            y = y + G(i).* exp(1i*m(i)*2*pi/T*t); %w0 = 2*pi/T0;
    end
    y = real(y)/N;
    b = 1;%4.1 K=8 works best
    d = .08; kappa = 3.9e-1;
       [tnIdx,yInt] = iafTEM(y,dt,b,d,kappa);
    tn = t(tnIdx); Ntn = length(tn);
    disp(Ntn)
    yDel = -b*diff(tn) + kappa*d;
    K = Kmax; 
    w0 = 2*pi/T;
F = exp(1j*w0*tn(2:end)'*(-K:K)) - exp(1j*w0*tn(1:end-1)'*(-K:K));
F(:,K+1) = tn(2:end) - tn(1:end-1);
s = T./(1j*2*pi*(-K:K)); s(K+1) = 1;
S = diag(s);

ytnHat = pinv(F*S)*yDel';
ytnHat = ytnHat'*N;


spectrum = conj(ytnHat(K+1:end));

%spectrum = conj(G(K:end)); 
%pectrum = conj(G); 
K = 10;%9; %K=8
spectrum = cadzow(spectrum,K,inf)';
swce = T*eye(length(ytnHat))\ytnHat';
swce = swce'; 

% wl = ann_filt(swce,K);
% estTaul = sort(-T*wl/(2*pi));
% 
% % Estimation of amplitudes
% E = exp(-1j*2*pi*(-length(ytnHat)/2:length(ytnHat)/2-1)'*estTaul'/T);
% estAl = real(pinv(E)*swce(:));

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
%uk = esprit(spectrum,K);
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
rk = fliplr(sort(rk2)); 
t = 0:1/N:1-1/N;
%T = 3;%tau 
ro = 2*K/T;
rk = sort(rk); 
ck = real(ck)/N;%0.16;
% ck(6) = ck(6)-0.016;
% ck(4) = ck(4)+0.05;
rk = rk+0.01; 
%ck(6)= ck(2)-0.04446;
    signal22 = 0;
     
    for k = 1:1:K
        signal22 = signal22+time_eval(ck,rk,tk,t,k);
    end
    %signal22 = circshift(signal22,20); 
    signal22 = signal22./max(signal22); 
signal2(idx+1:idx+2000)= signal22; 
%signal2(idx+1:idx+2000) = max(x_sig)*signal22/max(signal22);
  rmse = mean(sqrt(abs(x_sig-signal2(idx+1:idx+2000)).^2));
  RMSE = [RMSE rmse]; 
%plot(t,x_sig,'b',linewidth = 1); hold on; plot(t,(signal2(idx+1:idx+1200)),'--r',linewidth=1);grid on;legend('og','est');
    end 
L= signal2>1;
data2 = signal2; 
data2(L) = 0; 
   est = [est;data2/max(data2)]; 
   ref = [ref;x/max(x)];
   RMSE_all = [RMSE_all;RMSE]; 
end
L= est>1;
data2 = est; 
data2(L) = 0; 
save('est.mat','data2'); 
save('ref.mat','ref');
end