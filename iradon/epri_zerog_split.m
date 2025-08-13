function [data, zerog_data, data_idx, pars, pars_out] = ...
  epri_zerog_split(data_and_baseline, service_idx, pars)

ZEROG_INDEX = -1;
BASELINE_INDEX = 0;

ZGidx = service_idx == ZEROG_INDEX;
NOTZGidx = ~ZGidx;

data_idx = service_idx(NOTZGidx);

data = data_and_baseline(:, NOTZGidx);
zerog_data = data_and_baseline(:, ZGidx);

pars_out.G  = pars.G(ZGidx, :);
pars.G = pars.G(NOTZGidx, :);
pars.nTrace = size(data, 2);

if isfield(pars, 'Gexp')
 pars_out.Gexp        = pars.Gexp(ZGidx, :);
 pars.Gexp = pars.Gexp(NOTZGidx, :);
end

if safeget(pars, 'UseBaseline', true) == true
  BLindex = zeros(size(ZGidx));
  BL = find(service_idx == BASELINE_INDEX, 1, 'first');
  for ii=1:length(ZGidx)
    if ZGidx(ii), BLindex(ii)=BL; end
    if service_idx(ii) == BASELINE_INDEX, BL = ii; end
  end
  BLindex = BLindex(service_idx == ZEROG_INDEX);
  
  for ii=1:size(zerog_data,2)
    zerog_data(:,ii) = zerog_data(:,ii) - data_and_baseline(:,BLindex(ii));
  end
end

% Pulse methodologies
if isfield(pars, 'Offset')
  if length(pars.Offset) ~= length(ZGidx), pars.Offset = zeros(size(ZGidx)); end
  pars_out.Offset = pars.Offset(ZGidx);
  pars.Offset = pars.Offset(NOTZGidx);
end

% CW methodologies
if isfield(pars, 'UnitSweep')
  pars_out.UnitSweep = pars.UnitSweep(ZGidx);
  pars.UnitSweep = pars.UnitSweep(NOTZGidx);
end