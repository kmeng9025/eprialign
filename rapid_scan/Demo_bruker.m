clc;clear all
%%
par.hm=12.340;        % pp modulation amplitude [G]
par.Vm=20962.17;     % Scanning frequency  [Hz]
par.fwhm=.5;       % Post-filtering with Gaussian profile
par.up=1;            %
par.ph=10;
par.fp=-0.0145; % in % of full cycle
par.fig=1;           % ON/OFF figure; 
par.method='fast';   % 'default'
par.msg=1;           % ON/OFF message
bruker_filename='dm109405.dsc';
par.N_iter=3;
%% Load Rapid Scan
GN='./'; % data folder
totalRS=[GN bruker_filename];

if exist(totalRS);     disp(totalRS);
    [t par.rs p]=eprload(totalRS); % load Bruker data file
else
    disp(' file location is not correct'); break
end

%% Deconvolution
if par.fig==0; close all; end;
par.dt=(t(2)-t(1))*1E-9; % converts nanoseconds into second

tic
[h A B rs_out bg_out]=sinDecoBG(par);
deconvolutionTime_=toc
 % plot(h,A+B)


