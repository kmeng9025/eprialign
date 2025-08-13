% RADON_PHANTOM  Digital phantom generator
%  = struct('PhantomShape', 'Spherical', 'nBins', 128, 'matSizeCm', 4.24, 'r', 2.5/2, 'offset', [.5,.5,0]);
% [Ph_image] = RADON_PHANTOM(sphere);
%
%   See also RADON_C2D_SPHERE, RADON_C2D_CUBE.

% Author: Jayasai Rajagopal, Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, 2015
% Contact: jrajagopal@uchicago.edu

function [Ph_image] = radon_phantom( phantom )
% Generates a digital phantom for discrete randon transforms
%
%   for all inputs
%       size - size of mesh
%       nBins - number of pixels
%
%   for type sphere
%       r - radius of sphere (cm)
%       offset - offset from center in (x,y,z) form (cm)
%

r = safeget(phantom, 'r', 5);
l = safeget(phantom, 'l', 1);
a = safeget(phantom, 'a', 2);
offset = safeget(phantom, 'offset', [0,0,0]);
size = safeget(phantom, 'matSizeCm', 10);
nBins = safeget(phantom, 'nBins', 64);

x=linspace(-size/2,size/2,nBins);
y=linspace(-size/2,size/2,nBins);
z=linspace(-size/2,size/2,nBins);

[X,Y,Z]=meshgrid(x,y,z);

switch safeget(phantom, 'type', 'sphere')
  case 'circle'
    Ph_image=zeros(nBins,nBins,1);
    Ph_image((X(:,:,1)-offset(1)).^2+(Y(:,:,1)-offset(2)).^2<=r^2)=1;
  case 'sphere'
    Ph_image=zeros(nBins,nBins,nBins);
    Ph_image((X-offset(1)).^2+(Y-offset(2)).^2+(Z-offset(3)).^2<=r^2)=1;
  case 'cylinder'
    Ph_image=zeros(nBins,nBins,nBins);
    L1 = -l/2;
    L2 = +l/2;
    Ph_image((X-offset(1)).^2+(Y-offset(2)).^2<=r^2 & (Z-offset(3)) > L1 & (Z-offset(3)) < L2)=1;
  case 'cube'
    Ph_image=zeros(nBins,nBins,nBins);
    Ph_image(abs(X-offset(1)) <= a/2 & abs(Y-offset(2)) <= a/2 & abs(Z-offset(3)) <= a/2)=1;
  otherwise
end

