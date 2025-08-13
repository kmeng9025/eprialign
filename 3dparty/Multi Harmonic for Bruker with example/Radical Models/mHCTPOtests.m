clear all; clc;
T2=2.2500e-6;
sweep=3;
[x, y]=mHCTPO(sweep,T2);
plot(x,y);

% fwhm=2/(gamma*T2); % FWHM
% dh=50/M*fwhm;
% %%
% x0=(-M/2:M/2)*dh;
% shift=round(AH/2/dh)
% x=(-9:1:9)*0.041;                     % HF position with offset =0;
% y=[.000662 .002298 .006458 .01519 .030551 .053278 0.081386 0.10966 0.1309195 0.138847 0.1309195 0.10966 0.081386 0.0532778 0.0305509 0.015193  0.00646 0.002298 0.000662];  % HF intensities
% Y=interp1(x,y,x0); Y(isnan(Y))=0; 
% envelope=(circshift(Y',shift)+circshift(Y',-shift))';
% plot(x0,envelope);
% % + Lorentzian broadening
% diff=1;
% lor = lorentzian(x0,0,fwhm,diff);
% res=conv(envelope,lor,'same');
% plot(x0,res);


% break
% AH=0.48+0.;                              
% x1=x-AH/2;
% x2=x+AH/2;
% X=[x1 x2];
% Y=[y y];
% plot(X,Y,'-.');
% 
% %%  Lorentzian 
% diff=1;
% 
% M=1000;
% XX=[];
% 
% for k=1:length(x); 
% x0=(-M/2:M/2)*5/M*fwhm;
% 
% plot(x,y); axis tight
% 
% 
