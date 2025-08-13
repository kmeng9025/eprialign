% function [idx_roi, axis_roi, proj_xyz] = epr_GetROIbins(FOV, ROI, nbins, xyz)
% FOV - image field of view
% ROI - image region of interest
% nbins - number of bins in the image
% xyz - some point in the image bin coordianates
% idx_roi.x//.y//.z  - index of bins of ROI
% axis_roi.x//.y//.z - axis of ROI
% proj_xyz - some point in the image coordianates
% Example: 
%    [idx_roi, axis_roi, proj_xyz] = epr_GetROIbins(4, 3, 64, [32,32,32]);

function [idx_roi, axis_roi, proj_xyz] = epr_GetROIbins(FOV, ROI, nbins, xyz)

nDim = length(nbins);

if length(FOV) ~= nDim,
  FOV = FOV(1)*ones(1, nDim);
  ROI = ROI(1)*ones(1, nDim);
end

a = nbins.*ROI./FOV;
fld_names = ['x','y','z','b'];
idx_roi = []; proj_xyz = ones(3,1);

for ii=1:nDim
  startN=(nbins(ii)-a(ii))/2+1;
  endN=nbins(ii)-startN+1;
  idx_tmp =fix(round(startN):round(endN));
  idx_roi.(fld_names(ii)) = idx_tmp(idx_tmp > 0 & idx_tmp <= nbins(ii))';
  full_axis = FOV(ii)*(-0.5:1/(nbins(ii)-1):0.5)';
  if exist('xyz', 'var') && ~isempty(xyz), proj_xyz(ii) = full_axis(xyz(ii)); end
  axis_roi.(fld_names(ii)) = full_axis(idx_roi.(fld_names(ii)));
end

if nDim == 2
  axis_roi.z = [];
  idx_roi.z = [];
end
