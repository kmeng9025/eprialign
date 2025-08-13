%RADON_D2D - 3D radon transformation
% [pr, r] = RADON_D2D_SINGLE4(M, marix_size, radon_pars)
% M - 3D matrix of an object
% marix_size - matrix size in [cm]
% radon_radon_pars - structure of parameters
%   x,y,z arrays of unit vector projections on coordinate axis
%   nSubBins - each voxel will be brocken into 2^3 subvoxels prior to projetion 
%
%   See also RADON_ANGLE2XYZ.

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, 2013
% Contact: epri.uchicago.edu

function [P, r] = radon_d2d_single4(M, phantom, fbp_pars, pars_ext)

all_sin = sin(pars_ext.alpha);
all_cos = cos(pars_ext.alpha);
alpha_idx = pars_ext.k;
% Radon transformation parameters
radon_pars.un = [pars_ext.unx.*all_sin, pars_ext.uny.*all_sin, pars_ext.unz.*all_sin];            % unit vectors of the gradients
radon_pars.size  = phantom.matSizeCm;  % projection spatial support (cm)
radon_pars.nBins = 64;                 % length of the projection spatial support

sigma = safeget(phantom, 'lw', 5);
center = safeget(phantom, 'center', 0);
phase  = safeget(phantom, 'phase', 0) * pi/180;

matSizeG = safeget(phantom, 'matSizeG', 1.024*sqrt(2));

[P] = radon_d2d(M, phantom.matSizeCm, radon_pars);
nBins4 = size(P,1);

iidx = unique(alpha_idx);

for ii=1:length(iidx)
  idx = find(alpha_idx == iidx(ii));

  x = linspace(-matSizeG/2,matSizeG/2,nBins4)/mean(all_cos(idx));
  switch upper(safeget(phantom, 'spectrum', 'LORENTZIAN'))
    case 'LORENTZIAN'
      spectrum = real(exp(-1i*phase)*epr_Lorentzian(x, center, sigma(1)));
    otherwise
  end
  
  %   convolve with spectrim
  for jj=1:length(idx)
    P(:,idx(jj)) = conv(P(:,idx(jj)), spectrum, 'same');
%     figure(50); plot(x,P(:,idx(ii))*3000,x,spectrum,x,conv(P(:,idx(ii)), spectrum, 'same')*100)
  end
end

return




