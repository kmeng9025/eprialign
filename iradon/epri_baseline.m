% pars = epri_baseline(opt)
% opt.baseline  - type of b/l corection
%       BEFORE
%       AFTER
%       BEFORE_AFTER
%       EVERY_N  - b/l recorded for every n (= opt.bl_n) projection
% opt.bl_n      - parameter used in EVERY_N baseline correction

function pars = epri_baseline(pars, opt)

BASELINE_INDEX = 0;
ZERO_GRADIENT_INDEX = -1;

baseline = upper(safeget(opt,'baseline','none'));
zerogradient = upper(safeget(opt,'zero_gradient','none'));

switch baseline
  case 'BEFORE'
    pars.service_idx   = [BASELINE_INDEX;pars.service_idx];
  case 'AFTER'
    pars.service_idx   = [pars.service_idx;BASELINE_INDEX];
  case 'BEFORE_AFTER'
    pars.service_idx   = [BASELINE_INDEX;pars.service_idx;BASELINE_INDEX];
  case 'EVERY_N'
    every_n = safeget(opt, 'bl_n', 100);
    new_prj = pars.nTrace*(1 + 1/every_n);
    for ii=0:(every_n+1):new_prj-1
      pars.service_idx   = [pars.service_idx(1:ii);BASELINE_INDEX;pars.service_idx(ii+1:end)];
    end
    pars.service_idx   = [pars.service_idx;BASELINE_INDEX];
end

% Gradient and Sweep tables
pars.nTrace = length(pars.service_idx);
b_idx   = pars.service_idx == BASELINE_INDEX;
nb_idx  = ~b_idx;
G = pars.G; pars.G = zeros(pars.nTrace, 3); pars.G(nb_idx, :) = G;
Sweep = pars.UnitSweep; pars.UnitSweep = zeros(pars.nTrace, 1); pars.UnitSweep(nb_idx) = Sweep;

% Determine gradients during baseline and set it to next projection
% gradient
switch 'next_projection'
  case 'next_projection'
    b_idx_next   = find(b_idx);
    b_idx_next = b_idx_next+1; b_idx_next(b_idx_next > pars.nTrace) = pars.nTrace;
    
    pars.G(b_idx, :) = pars.G(b_idx_next, :);
    pars.UnitSweep(b_idx) = pars.UnitSweep(b_idx_next);
    
    % Check last element and set it to previous
    if pars.service_idx(end) == BASELINE_INDEX
      pars.G(end,:) = pars.G(end-1,:);
      pars.UnitSweep(end) = pars.UnitSweep(end-1);
    end
end

Sweep = pars.UnitSweep;
sw_next = mean(Sweep);
switch zerogradient
  case 'BEFORE'
    unique_sweeps = unique(pars.UnitSweep);
    for ii=1:length(unique_sweeps)
      idx = find(Sweep == unique_sweeps(ii), 1, 'first');
      pars.service_idx   = [pars.service_idx(1:idx-1);ZERO_GRADIENT_INDEX;pars.service_idx(idx:end)];
      Sweep   = [Sweep(1:idx-1);sw_next;Sweep(idx:end)];
    end
  case 'AFTER'
    unique_sweeps = unique(pars.UnitSweep);
    for ii=1:length(unique_sweeps)
      idx = find(Sweep == unique_sweeps(ii), 1, 'last');
      pars.service_idx   = [pars.service_idx(1:idx);ZERO_GRADIENT_INDEX;pars.service_idx(idx+1:end)];
      Sweep   = [Sweep(1:idx);Sweep(idx);Sweep(idx+1:end)];
    end
  case 'BEFORE_AFTER'
    unique_sweeps = unique(pars.UnitSweep);
    for ii=1:length(unique_sweeps)
      idx = find(Sweep == unique_sweeps(ii), 1, 'first');
      pars.service_idx   = [pars.service_idx(1:idx-1);ZERO_GRADIENT_INDEX;pars.service_idx(idx:end)];
      Sweep   = [Sweep(1:idx-1);Sweep(idx);Sweep(idx:end)];
      idx = find(Sweep == unique_sweeps(ii), 1, 'last');
      pars.service_idx   = [pars.service_idx(1:idx);ZERO_GRADIENT_INDEX;pars.service_idx(idx+1:end)];
      Sweep   = [Sweep(1:idx);Sweep(idx);Sweep(idx+1:end)];
    end
  case 'EVERY_N'
    every_n = safeget(opt, 'zg_n', 100);
    new_prj = length(pars.service_idx)*(1 + 1/every_n);
    for ii=0:(every_n+1):new_prj-1
      pars.service_idx   = [pars.service_idx(1:ii);ZERO_GRADIENT_INDEX;pars.service_idx(ii+1:end)];
      Sweep   = [Sweep(1:ii);sw_next;Sweep(ii+1:end)];
    end
    pars.service_idx   = [pars.service_idx;ZERO_GRADIENT_INDEX];
    Sweep   = [Sweep;sw_next];
end

% Gradients (set to zero) and Sweep tables
pars.nTrace = length(pars.service_idx);
b_idx   = pars.service_idx == ZERO_GRADIENT_INDEX;
nb_idx  = ~b_idx;
G = pars.G; pars.G = zeros(pars.nTrace, 3); pars.G(nb_idx, :) = G;
pars.UnitSweep = Sweep;
