function [success_rate_3bpm,success_rate_2bpm,PCC,MAE,RMSE] = GT_comparison_TEM(type,method,x,y,Radar_time_vec,win_dur,intvl_dur)

%% HR_GT_comparison: x =  radar estimates, y = reference estimates (padded FFT/findpeaks)
% % figure; hold on
% % plot(Radar_time_vec,y,'b','linewidth',1) 
% % plot(Radar_time_vec,x,'r','linewidth',1)
% % % title({['Comparison to Ground-Truth - ',method];['Computation windows duration: ',num2str(win_dur),' s'];['Time interval: ',num2str(intvl_dur),' s']});
% % xlabel('Time (Sec)'); 
% % legend('ECG reference',method) 
% % if type =='RR'
% %     ylabel('RR (b.p.m.)');
% %     ylim([0 30])
% % else
% %     ylabel('HR (b.p.m.)');
% %     ylim([40 120])
% % end
TP_3bpm = find(abs(x-y)<=3);
success_rate_3bpm = (length(TP_3bpm)/length(x))*100;

TP_2bpm = find(abs(x-y)<=2);
success_rate_2bpm = (length(TP_2bpm)/length(x))*100;

R = corrcoef([x y]);
PCC = abs(R(1,2))*100;

MAE = mean(abs(x-y));

RMSE = sqrt(mean((x-y).^2)); 
end

