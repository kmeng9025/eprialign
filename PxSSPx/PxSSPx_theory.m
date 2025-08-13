function peaks = PxSSPx_theory(x,y,the_state,opts)

peaks.x = x(:);

OFF = safeget(opts, 'CF', 0);
A   = safeget(opts, 'A', 20);
J   = safeget(opts, 'J', 100); % [MHz]
LW   = safeget(opts, 'LW', [0.5 0.25]);

bi.S = [1/2 1/2];
bi.g = [2.003 2.003];

bi.A = [A 0; 0 A]*2.802; % MHz
bi.ee = J;  %  assuming H = +J*S1*S2
bi.Nucs = '15N, 15N'; % or whatever you have
bi.lwpp = LW; % mT

Exp.mwFreq = 0.25; % GHz
offset = Exp.mwFreq/28.02E-3 + OFF*0.1;
Exp.Range = offset+0.1*[min(peaks.x) max(peaks.x)]; % mT
% Exp.Range = [4.5 13]; % mT
Exp.Harmonic = 0;

[~, peaks.y] = pepper(bi,Exp); hold on
peaks.y =  interp1(peaks.y(:),linspace(1,length(peaks.y),length(y)),'linear',0);
peaks.simint = sum( peaks.y(:));
peaks.y =  peaks.y(:) / peaks.simint;
peaks.amp = sum(y(:).*peaks.y) / sum(peaks.y.*peaks.y);

if isfield(opts, 'fig') && ~isempty(opts.fig)
  figure(opts.fig); clf;
  plot(x, y); hold on
  plot(peaks.x, peaks.y*peaks.amp); hold on
end