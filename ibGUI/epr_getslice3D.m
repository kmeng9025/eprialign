% EPR_GETSLICE3D  return three orthogonal slices of a 3D matrix
% function [yx, zx, yz]=epr_getslice3D(data, proj)
% data - 3D matrix
% proj - [x,y,z] slice coordinates or []

% Boris Epel (c) 2007-2010
% University of Chicago
% bepel@uchicago.edu

function [yx, zx, yz]=epr_getslice3D(data, proj, idx_x, idx_y, idx_z)

if nargin<2
  help epr_getslice3D;
  return
end

if ~exist('idx_x', 'var'), idx_x = 1:size(data,2); end
if ~exist('idx_y', 'var'), idx_y = 1:size(data,1); end
if ~exist('idx_z', 'var'), idx_z = 1:size(data,3); end

if isempty(proj)
  proj = fix(0.5*size(data));
end

yx = squeeze(data(idx_x,idx_y,proj(3)));
zx = squeeze(data(idx_x,proj(1),idx_z));
yz = squeeze(data(proj(2),idx_y,idx_z))';