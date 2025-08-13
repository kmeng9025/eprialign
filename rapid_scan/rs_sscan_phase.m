% RS_SSCAN_PHASE  sinosoidal scan deconvolution (interface to sinDecoBG)
% [x_ss,ss,scan_phase] = RS_SSCAN_PHASE(rs, sweep_pp, frequency, dwell_time, par_struct)
% rs         - Data, time along columns [array, 2D] 
% sweep_pp   - Sinusoidal scan amplitude, peak to peak [float, in G]
% frequency  - Frequency of field scan [float, in Hz]
% dwell_time - Time step of data [float, in s]
% par_struct - Additional parameters [structure]
%   [].field_scan_phase     - Scan phase [float, -360 to 360]
%   [].scan_phase_algorithm - Scan phase algorithm [string (auto) | manual]
%   [].up_down              - Sinus polarity [string (up_down) | down_up]
%   [].Fconv                - Filter adjusted to line width [float, in G]
%   [].display              - Level of display [string (off) | message | figure | all] 
% x_ss/ss    - Rapid scan field/spectra
% scan_phase - Phase of the scan
% See also RS_SFBP, RS_SDECONVOLVE, RS_GET_FIELD_PHASE.

% Author: Boris Epel
% For Rapid Scan code authors see inside sinDecoBG
% Center for EPR imaging in vivo physiology
% University of Chicago, 2013
% Contact: epri.uchicago.edu

function [x_ss,AB,scan_phase] = rs_sscan_phase(rs, sweep, Fm, dt, pars)

sz = size(rs);

par.dt = dt;
par.Vm = Fm;
par.hm = sweep;

if strcmp(safeget(pars, 'scan_phase_algorithm', 'auto'), 'auto')
  par.N_iter = 3;
else
  par.N_iter = 1;
end

% Average projections
par.fwhm = safeget(pars, 'Fconv', 0.1); % Post-filtering with Gaussian profile
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

par.ph = safeget(pars, 'data_phase', 0);
par.bw = 0;
par.fp = safeget(pars, 'field_scan_phase', 0);
par.method='fast';

par_im = par;
par_im.ph = safeget(pars, 'data_phase', 0);
par_im.N_iter = 1;

phase_absolute=zeros(1, sz(2));
for ii=1:sz(2)
    par.rs = rs(:,ii);
    [h, A, B, fp_corr_total]=sinDecoBG(par);
    phase_absolute(ii)=par.fp-fp_corr_total;
    x_ss = h;
    par_im.rs = rs(:,ii);
    par_im.fp = phase_absolute(ii);
    [h, A1, B1]=sinDecoBG(par_im);
    AB(:,ii)=A+B+1i*(A1+B1);
end
scan_phase = mean(phase_absolute);
