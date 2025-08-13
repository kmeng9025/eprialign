% RS_SDECONVOLVE  sinosoidal scan deconvolution (interface to sinDecoBG)
% [x_ss,ss] = RS_SDECONVOLVE(rs, sweep_pp, frequency, dwell_time, par_struct);
% rs         - Data, time along columns [array, 2D] 
% sweep_pp   - Sinusoidal scan amplitude, peak to peak [float, in G]
% frequency  - Frequency of field scan [float, in Hz]
% dwell_time - Time step of data [float, in s]
% par_struct - Additional parameters [structure]
%   [].field_scan_phase     - Scan phase [float, -360 to 360]
%   [].up_down              - Sinus polarity [string (up_down) | down_up]
%   [].display              - Level of display [string (off) | message | figure | all] 
% x_ss/ss    - Rapid scan field/spectra
% See also RS_SFBP, RS_SSCAN_PHASE, RS_GET_FIELD_PHASE.

% Author: Boris Epel
% For Rapid Scan code authors see inside sinDecoBG
% Center for EPR imaging in vivo physiology
% University of Chicago, 2013
% Contact: epri.uchicago.edu

function [x_ss, AB]=rs_sdeconvolve(rs, sweep, Fm, dt, in_pars)
sz = size(rs);

par.fp=in_pars.field_scan_phase;
par.dt = dt;
par.Vm = Fm;
par.hm = sweep;
par.N_iter = safeget(in_pars, 'N_iter', 3);

% Average projections
par.fwhm = safeget(in_pars, 'Fconv', 0.001); % Post-filtering with Gaussian profile
up_down = safeget(in_pars, 'up_down', 'up_down');
if strcmp(up_down, 'down_up')
  par.up = -1;   %
else
  par.up = 1;   %
end
display_level =  safeget(in_pars, 'display', 'off');
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

par.ph=90+safeget(in_pars, 'data_phase', 0);
par.bw = 0;
par.method='fast';
tic
%%
for ii=1:sz(2)
    par.rs = rs(:,ii);
    [x_ss, A, B]=sinDecoBG(par);
    AB(:,ii)=A+B;
end

% disp(fp_corr_total); % it is correction NOT absolute phase


