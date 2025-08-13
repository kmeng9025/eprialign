% RS_SDECONVOLVE  sinosoidal scan deconvolution (interface to sinDecoBG)
% [x_ss,ss] = RS_SDECONVOLVE(rs, sweep_pp, frequency, dwell_time, par_struct);
% rs         - Data, time along columns [array, 2D] 
% sweep_pp   - Sinusoidal scan amplitude, peak to peak [float, in G]
% frequency  - Frequency of field scan [float, in Hz]
% dwell_time - Time step of data [float, in s]
% par_struct - Additional parameters [structure]
%   [].field_scan_phase     - Scan phase [float, -360 to 360]
%   [].up_down              - Sinus polarity [string (up_down) | down_up]
%   [].display              - Level of display [string off | message | figure | all] 
% x_ss/ss    - Rapid scan field/spectra
% See also RS_SFBP, RS_SSCAN_PHASE.

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, 2013
% Contact: epri.uchicago.edu

function [x_ss,ss]=rs_sinus_scan(rs, sweep, Fm, dt, pars)

Ntr = size(rs, 2);

par.dt = dt;
par.Vm = Fm;
par.hm = sweep;

% Average projections
par.fwhm = safeget(pars, 'Fconv', 0.001); % Post-filtering with Gaussian profile
up_down = safeget(pars, 'up_down', 'up_down');
if strcmp(up_down, 'down_up')
  par.up = -1;   %
else
  par.up = 1;   %
end
display_level =  safeget(pars, 'display', 'off');
switch display_level
  case 'all'
    par.fig=1;             % ON/OFF figure
    par.msg=1;             % ON/OFF message
  case 'message'
    par.fig=0;             % ON/OFF figure
    par.msg=1;             % ON/OFF message
  case 'figure'
    par.fig=1;             % ON/OFF figure
    par.msg=0;             % ON/OFF message
  otherwise 
    par.fig=0;             % ON/OFF figure
    par.msg=0;             % ON/OFF message
end

par.ph=90;
par.bw = 0;
par.fp=pars.field_scan_phase/360;
par.method='fast';

tic
gamma=1.7608e7;
g2f=gamma/(2*pi); % = 2.8024e6

M=length(rs);
t=(0:(M-1))*dt;       % Time vector ( raw data)
Vmax=g2f*sweep;       % Max possible RS signal frequency
Ns=2*ceil(Vmax/Fm);   % Min number of points in the frequency domain(= the time domain)
P=1/Fm;               % Scan period
ts=(0:(Ns-1))*(P/Ns); % time vector for final filtering & interpolation

Fmax=1/(2*dt);        % Max frequency to be sampled without aliasimg
ratio=Vmax/Fmax;      % sampling ratio must be <1
Wm=2*pi*Fm;

Nc=round(1/(dt*Fm));      % Points per period
Nfc=floor(M/Nc);           % Number of full cycles

for tr = 1:Ntr
  par.rs = rs(:,tr);
  
  rs_i=InterleavingCycles(t,ts,rs(:,tr)',Fm,Nfc,Ns,Wm,par.method);
  
  rs_i=circshift(rs_i, [0, round(pars.field_scan_phase/360*length(rs_i))]);

  L = length(rs_i);
  x_cos = -sweep*cos(linspace(0,2*pi*(1 - 1/L),L))/2;
  Lhalf = fix(L/2);
  x_lin = sweep*linspace(-1,1,Lhalf)/2;
  x_ss = x_lin;
  ss(1:Lhalf,tr) = interp1(x_cos(1:Lhalf), rs_i(1:Lhalf), x_lin) ...
            + interp1(x_cos(Lhalf+1:L), rs_i(Lhalf+1:L), x_lin);
end
toc;

