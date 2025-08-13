clc; clear all
%%  Solution of Bloch Equations
par.hm=5;     %  pp modulation amplitude [G]
par.dH=0.5;      %  offset from central field position
par.T1=.5e-6;  % T1 [s]
par.T2=par.T1; % T2 [s]
par.Vm=25e3;   %  scanning frequency  [Hz]
par.B1=.005;   % B1 field [G]

%%
[t, mx, my, mz]=blochSin(par);
par.dt=t(2)-t(1);
rs=mx+1i*my;  % Complex RS signal
par.rs=[rs rs(1)];
%%

%% Deconvolution

%% ------------------------------------------------------
par.fwhm=0.002; % Post-filtering with Gaussian profile
par.bw=10e6; % >= Resonator BW
par.up=1;    %
par.ph=0;
par.fp=0.5;
par.fig=1;
[h A B]=sinDecoBG(par);
% %% To Play with adding noise and background
% bg=cos(2*pi*par.Vm*t+pi/3)+1i*sin(2*pi*par.Vm*t+pi/4)+1+1i; % background
% bg=bg+cos(4*pi*par.Vm*t-pi*2/3)+1i*sin(4*pi*par.Vm*t-pi/4)+1+1i; % background
% noise=randn(size(bg))/10;
%%
% playWith=0.01;  % make it =0 if not in the mood to play
%par.rs=par.rs+playWith*(bg+noise);



