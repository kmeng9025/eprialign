% To demonstrate the procedure data to be deconvoluted are
% generated with the Bloch equations. Experimental data could
% be read into array us.
clc;clear all

ReGenerateData=0  % '1' to solve Bloch equations with new parameters 
                  % '0' to explore generated solution 
if ReGenerateData==1
    %%  Solution of Bloch Equations
    % with the following parameters:
    par.hm=10;      % pp modulation amplitude [G]
    par.dH=-0.;      % offset from central field position
    par.T1=2.5e-6;  % T1 [s]
    par.T2=par.T1; % T2 [s]
    par.Vm=7001;   %  scanning frequency  [Hz]
    par.B1=.005;   % B1 field [G]
    M=10000;       % Number of points per period
    P=1/par.Vm;    % Period
    par.dt=1.11*P/M;    % Time per point in simulation
    Nc=3.53;          % Number of full cycles
    t=0:par.dt:(Nc*P-par.dt); % Time array for two full cycles
    err=1e-11;       % Error tolerance
    options = odeset('RelTol',err,'AbsTol',[err err err]);
    % Solution of Bloch Eqs with above pars
    [Tau,Y]=ode45(@RSS,t,[0 0 1],options,par);
    mx=-Y(:,1)';
    my=-Y(:,2)';
    par.rs=mx+1i*my;  % Complex RS signal
    save tmp
else
    load tmp
end
%% test
L=1;
par.rs=par.rs(1:L:end);
par.dt=par.dt*L;
%%

%% Deconvolution

%% ------------------------------------------------------
par.fwhm=.01; % Post-filtering with Gaussian profile
%par.bw=10e6; % >= Resonator BW
par.up=1;    % 
par.ph=30;
par.fp=-16;
par.fig=1;             % ON/OFF figure
par.method='fast';
par.msg=0;             % ON/OFF message
par.N_iter=3;
%% To Play with adding noise and background
bg=cos(2*pi*par.Vm*t+pi/3)+1i*sin(2*pi*par.Vm*t+pi/4)+1+1i; % background 
bg=bg+cos(4*pi*par.Vm*t-pi*2/3)+1i*sin(4*pi*par.Vm*t-pi/4)+1+1i; % background 
noise=randn(size(bg))+1i*randn(size(bg));
%%
playWith=0.1;  % make it =0 if not in the mood to play 
par.rs=par.rs+playWith*(bg+noise/100);


[h A B fp_corr rs_out bg_out ]=sinDecoBG(par);





