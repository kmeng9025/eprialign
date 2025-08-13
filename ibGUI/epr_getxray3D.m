% epr_getxray3D  return three orthogonal slices projected through matrix
% function [yx, zx, yz]=epr_getxray3D(data, proj)
% data - 3D matrix
% proj - [x,y,z] slice coordinates or []

% Boris Epel (c) 2007-2010
% University of Chicago
% bepel@uchicago.edu

function [yx, zx, yz]=epr_getxray3D(data, proj, idx_x, idx_y, idx_z)

if nargin<2
  help epr_getslice3D;
  return
end

if ~exist('idx_x', 'var'), idx_x = 1:size(data,2); end
if ~exist('idx_y', 'var'), idx_y = 1:size(data,1); end
if ~exist('idx_z', 'var'), idx_z = 1:size(data,3); end

yx = squeeze(sum(data(idx_x,idx_y,:), 3));
zx = squeeze(sum(data(idx_x,:,idx_z), 2));
yz = squeeze(sum(data(:,idx_y,idx_z), 1))';