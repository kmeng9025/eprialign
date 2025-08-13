function fp_absolute = rs_scan_phase(rs, sweep, Fm, dt, pars)

Ntr = size(rs, 2);

par.dt = dt;
par.Vm = Fm;
par.hm = sweep;
par.N_iter = 3;

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
par.fp=pars.field_scan_phase;
par.method='fast';
tic

par.rs = rs;
[h, A, B, fp_corr_total, rs_out, bg_out]=sinDecoBG(par);
fp_absolute=par.fp-fp_corr_total;

