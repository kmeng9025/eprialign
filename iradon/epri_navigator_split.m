function [mat, mat_nav, out_idx, out_nav_idx, processed_info, bl_info] = ...
  epri_navigator_split(data_and_baseline, service_idx, processed_info)

NAVIGATOR_INDEX = 100;

NVidx = service_idx == NAVIGATOR_INDEX;
NOTNVidx = ~NVidx;

out_idx = service_idx(NOTNVidx);
out_nav_idx = service_idx(NOTNVidx);

mat = data_and_baseline(:, NOTNVidx);
mat_nav = data_and_baseline(:, NVidx);

bl_info.G = processed_info.G(NVidx, :);
processed_info.G = processed_info.G(NOTNVidx, :);

if isfield(processed_info, 'Gexp')
 bl_info.Gexp        = processed_info.Gexp(NVidx, :);
 processed_info.Gexp = processed_info.Gexp(NOTNVidx, :);
end

% Pulse methodologies
if isfield(processed_info, 'Offset')
  bl_info.Offset = processed_info.Offset(NVidx);
  processed_info.Offset = processed_info.Offset(NOTNVidx);
end

% CW methodologies
if isfield(processed_info, 'UnitSweep')
  bl_info.UnitSweep = processed_info.UnitSweep(NVidx);
  processed_info.UnitSweep = processed_info.UnitSweep(NOTNVidx);
end