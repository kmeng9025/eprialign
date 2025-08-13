function [data, baseline, data_idx, pars, pars_out] = ...
  epri_baseline_split(data_and_baseline, service_idx, pars)

BASELINE_INDEX = 0;

BLidx = service_idx == BASELINE_INDEX;
NOTBLidx = ~BLidx;

data_idx = service_idx(NOTBLidx);

data = data_and_baseline(:, NOTBLidx);
mat_bl_traces = data_and_baseline(:, BLidx);

pars_out.G  = pars.G(BLidx, :);
pars.G = pars.G(NOTBLidx, :);
pars.nTrace = size(data, 2);

if isfield(pars, 'Gexp')
 pars_out.Gexp        = pars.Gexp(BLidx, :);
 pars.Gexp = pars.Gexp(NOTBLidx, :);
end

% Pulse methodologies
if isfield(pars, 'Offset')
  if length(pars.Offset) ~= length(BLidx), pars.Offset = zeros(size(BLidx)); end
  pars_out.Offset = pars.Offset(BLidx);
  pars.Offset = pars.Offset(NOTBLidx);
end

% CW methodologies
if isfield(pars, 'UnitSweep')
  pars_out.UnitSweep = pars.UnitSweep(BLidx);
  pars.UnitSweep = pars.UnitSweep(NOTBLidx);
end

if isempty(find(BLidx, 1))
  baseline = [];
elseif   length(find(BLidx)) == 1
  baseline = repmat(mat_bl_traces, size(data, 2));
else
  baseline = zeros(size(data));
  pBLidx = find(BLidx);
  pNOTBLidx = find(NOTBLidx);
  for jj=1:size(mat_bl_traces, 1)
    baseline(jj,:) = interp1(pBLidx, mat_bl_traces(jj,:), pNOTBLidx,'spline');
  end
end