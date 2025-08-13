function [h res]=sinDeco(hm,Fm,tb,us,Ne,cut)
% hm  - pp modulation amplitude [G]
% Fm  - modulation frequency    [Hz]
% us  - Rs spectrum, full cycle, complex.
% tb -  time base (sampling period) for us
% Ne -  Number of half-periods for extension
% cut - location for adjucent linear extension [%]
%        5% means that sinusoidal part is 90% of hm.
%% sinusoidal deconvolution: WF=Om/2*sin(2*pi*Fm*t);
gamma=1.7608e7;
Om=gamma*hm;        % Mod amplitude [rad/s]
Wm=2*pi*Fm;         % Mod frequency [rad/s]
P=1/Fm;             % Period of the scan [s]
P2=P/2;             % half-period [s]
xc=acos(1-cut/100); % x-position for line extention [rad]
tc=xc/(2*pi)*P;     % time moment for line extention [s]
Wc=(Om/2)*(1-cut/100);     %y-position for line extention [rad]
slope=Fm*sin(Wm*tc)*Om/2;  %slope for line extention [rad/s]
Vmax=Wc/(2*pi)+(Ne*P2+tc)*slope; % max Frequency for driving function [Hz]
Hmax=Vmax*2*pi/gamma;            % Corresponding Magnetic field [G]
Ts=round(0.3/Vmax*1e10)/1e10;    % Sampling period for driving function [s]
Nc=round(P2/Ts/2)*2;          % points per half-period
i=0:(Nc*(2*Ne+1)-1);
t=i*Ts;                  % time vector
W=(-1)^Ne*cos(2*pi*Fm*t)*Om/2;
%% Build an Extended wavefrom, W, And Driving function, ui.
% Left-hand
M=length(t);
[a i2]=min(abs(t-(tc+Ne*P2)));
i1=i2-1;
slope=W(i1)-W(i2);
W(i1:-1:1)=(1:i1)*slope+W(i2);
% Right-hand
xxx=(2*Ne+1)*P2-(Ne)*P2-tc;
[a i1]=min(abs(t-xxx));
i2=i1+1;
slope=W(i2)-W(i1);
W(i2:M)=((i2:M)-i2)*slope+W(i2);
PH=cumsum(W)*Ts; % Phase waveform

%plot(t/P2,PH);
ui=exp(-1i*PH);  % Driving function, DF
%% RS specrum intepolation
tc=t(M/2);       % central time position for DF
N=length(us)     % N points in full cycle RS signal
tx=(0:(N/2-1))*tb;  % experimental time vector
tx=tx-mean(tx)+tc;
%plot(tx,imag(us));
%us=shiftv1(us,-6);
us1=us(1:N/2);
us1=interp1(tx,us1,t);% UP -scan
in=isnan(us1);
us1(in)=0;

us2=interp1(tx,us(N/2+1:N),t);% UP -scan
in=isnan(us2);
us2(in)=0;
%plot(t,imag(us));

%% Deconvololution
%ui=shiftv1(ui,100);
UI1=fftshift(fft(ui));
UI2=fftshift(fft(conj(ui)));
ud1=us1.*ui;
ud2=us2.*conj(ui);
UD1=fftshift(fft(ud1));
UD2=fftshift(fft(ud2));
UO1=fliplr(UD1./UI1);
UO2=fliplr(UD2./UI2);

%%  Results
dV=1/t(end);
h=(1:M)*dV*2*pi/gamma;
h=h-mean(h);
inx=abs(h)<=hm/2;
h=h(inx);
UO1=UO1(inx);
UO2=UO2(inx);

subplot(2,1,1);
plot(h,imag(UO1),h,imag(UO2));
title('up and down','FontSize',20);
axis tight;
subplot(2,1,2);
res=UO1+UO2;
plot(h,imag(res),h,real(res));
title('up + down','FontSize',20);
axis tight;