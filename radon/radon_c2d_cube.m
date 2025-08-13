% RADON_C2D_CUBE  analytic radon transformation of a sphere
% Phan = struct('PhantomShape', 'Spherical', 'nBins', 128, 'matSizeCm', 4.24, 'r', 2.5/2, 'offset', [.5,.5,0]);
% radon_pars = struct('Theta', theta, 'Phi', phi);
% [radon_pars.P, l, k] = RADON_C2D_CUBE(Phan, radon_pars);
%
%   See also RADON_C2D_SPHERE.

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, 2013
% Contact: epri.uchicago.edu
function [pr, r] = radon_c2d_cube(Phan, radon_pars)

addpath('z:\CenterMATLAB\3dparty\geom3d\geom3d');
addpath('z:\CenterMATLAB\3dparty\geom3d\meshes3d');
addpath('z:\CenterMATLAB\3dparty\geom2d\geom2d');
addpath('z:\CenterMATLAB\3dparty\geom2d\polygons2d');

dispstat('','init');

tic
psize = size(radon_pars.un);
nPrj = psize(1);
nI = radon_pars.nBins;

if isempty(safeget(radon_pars, 'l', []))
  nP = ceil(nI*sqrt(3));
  if mod(nP,2) == 0, nP = nP + 1; end
  r = [1:nP]' - (nP+1)/2;
else
  r = radon_pars.r(:);
  nP = length(r);
end
pr = zeros(nP,nPrj);

a = Phan.a/radon_pars.size*radon_pars.nBins;
[nodes, faces] = createCube;    % unit cube
nodes = (nodes - 0.5)*a;

for iP = 1:nPrj
  p = zeros(nP,1);
  for ii=1:nP
    v = radon_pars.un(iP,:);
    plane = createPlane(v*r(ii), v);
    poly = polyhedronSlice(nodes, faces, plane);
    if ~isempty(poly)
      poly2d = planePosition(poly, plane);
      p(ii) = abs(polygonArea(poly2d));
    end
  end
  dispstat(sprintf('Computing: %i (%i) - %4.2fs', iP, nPrj, toc),'timestamp');
  pr(:,iP) = p;
end
toc

% type of output: kspace vs normal space
output = upper(safeget(radon_pars, 'output', 'space'));

switch output
  case 'KSPACE'
    kspace = linspace(-1/radon_pars.size/2, 1/radon_pars.size/2, nP);
    if nargout == 1
      varargout{1} = fftshift(fft(pr));
    else
      varargout{1} = kspace;
      varargout{2} = fftshift(fft(pr));
    end
  otherwise
    if nargout == 1
      varargout{1} = pr;
    else
      varargout{1} = r;
      varargout{2} = pr;
    end
end