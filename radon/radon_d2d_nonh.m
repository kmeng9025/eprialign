%RADON_D2D - 3D radon transformation with inhomogenious field
% [pr, r] = RADON_D2D(M, marix_size, radon_pars)
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

function [P, r] = radon_d2d_nonh(M, B, marix_size, radon_pars)

if nargin  < 2, help radon_d2d; return; end

tic
dispstat('','init')
nPrj = size(radon_pars.un, 1);

% create distance space
nSize = size(M);
x=linspace(-nSize(1)/2,nSize(1)/2,nSize(1));
y=linspace(-nSize(2)/2,nSize(2)/2,nSize(2));
z=linspace(-nSize(3)/2,nSize(3)/2,nSize(3));
[X,Y,Z]=meshgrid(x,y,z);

nPts = safeget(radon_pars, 'nBins', 64);
CenterMatrix  = (nPts+1)/2;

P = zeros(nPts,nPrj);

r    = [X(:), Y(:), Z(:)];
data = M(:);

nVox         = size(r,1);
Conversion   = marix_size / radon_pars.size;
for iP = 1:nPrj
  p = zeros(nPts,1);
  rr = sum(B(:)+r .* repmat(radon_pars.un(iP,:), [nVox, 1]), 2)*Conversion + CenterMatrix;
  idx = rr >= 1 & rr <= nPts & data > 0.5;
  data1 = data(idx);
  rr1   = rr(idx);
  for jj=1:length(data1)
    p = incrementRadon(p, data1(jj), rr1(jj) );
  end
  P(:,iP) = p;
  dispstat(sprintf('Computing: %i (%i) - %4.2fs', iP, nPrj, toc),'timestamp');
end
P = P / nVox;
toc

function pr = incrementRadon(pr, a, r)
r1 = floor(r);
delta = r - floor(r);
pr(r1) = pr(r1) + a * (1.0 - delta);
pr(r1+1) = pr(r1+1) + a * delta;
