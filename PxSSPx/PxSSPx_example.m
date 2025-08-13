% This is the example of using PxSSPx processing toolbox
% Data location
pars1.path = 'V:\data\RapidScan\2015\150423\RS_Mouse_010_session_1\';
pars1.fidx = 4951:2:5075;
pars1.ftemplate = '%itrace_010_SESSION1.d01';

N = 1;

% Rapid scan data
pars1.field_scan_phase = -3.1;
pars1.display = 'off';
pars1.data_phase = 0;
pars1.up_down = 'up_down';
pars1.N_iter=1;

pars1.fig = 100+N; % set [] to suppress display

% Load data
data1 = PxSSPx_data(pars1);

% Extract cleaved nitroxide (PxS) line shape
opt.fig = N;
peaksBroken1 = PxSSPx_peak(data1.x, data1.y(:,60),'cleaved' ,opt)

% Two alternatives for generation of PxSSPx line shape
opt.CF = peaksBroken1.OFF+0.3;
opt.A  = 21; opt.J = 110;
opt.LW = 6.5;
% peaksPxSSPx1  = PxSSPx_peak(data1.x, data1.y(:,1),'PxSSPx' ,opt)
peaksPxSSPx1  = PxSSPx_theory(data1.x, data1.y(:,1),'PxSSPx' ,opt)

% Two alternatives to decompose line shape into PxSSPs and PxS components
res1 = PxSSPx_decompose(data1, peaksPxSSPx1, peaksBroken1); % using lineshapes
% res1 = PxSSPx_peak_kinetics(data1, peaksBroken1, opt);    % using peak amplitude
% res1.CLEAVED = res1.L1;
% res1.PxSSPx = res1.L2;

% fit kinetics
opt.fp = 1;
opt.fig = N;
fit = PxSSPX_fit(res1, 'all', opt)
%    k_F3: 332.8049    - cleavage kinetics from PxS build up
%    c_F3: 2.0831e+03  - bioreduction/clearance kinetics
%    k_F2: 619.5252    - cleavage kinetics from PxSSPx decay