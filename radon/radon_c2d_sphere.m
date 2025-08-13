% RADON_C2D_SPHERE  analytic Radon transformation of a sphere
% [P] = RADON_C2D_SPHERE(Phan, radon_pars);
% [r,P] = RADON_C2D_SPHERE(Phan, radon_pars);
% Phan - Phantom structure
%   r - sphere radius [cm]
%   offset - offset of sphere from center [cm]
%   nBins - length of array
% radon_pars - radon transformation parameters structure
%   x,y,z - vectors of unit vector coordinates
%   size - length of resulting projection [cm]
%
%   See also RADON_C2D_CUBE.

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, 2013
% Contact: epri.uchicago.edu

function varargout = radon_c2d_sphere(Phan, radon_pars)

% define or read projection ordinate r and number of points nP
if isempty(safeget(radon_pars, 'l', []))
  r = linspace(-radon_pars.size/2, radon_pars.size/2, radon_pars.nBins)';
  nP = radon_pars.nBins;
else
  r = radon_pars.l(:);
  nP = length(r);
end

psize = size(radon_pars.un);
nPrj = psize(1);

if psize(2) == 2 % 2D case  
  % circle offset from the center
  offset = safeget(Phan, 'offset', [0,0]);
  offset = offset(1:2);

  % analytic Radon transformation of a sphere
  off = sum(radon_pars.un.*repmat(offset(:), [1, nPrj])', 2);
  pr = pi*(Phan.r^2 - (repmat(r, [1,nPrj])-repmat(off, [1,nP])').^2);
  pr(pr<0) = 0;
  
else    % 3D case  
  % sphere offset from the center
  offset = safeget(Phan, 'offset', [0,0,0]);
  
  % analytic Radon transformation of a sphere
  off = sum(radon_pars.un.*repmat(offset(:), [1, nPrj])', 2);
  pr = pi*(Phan.r^2 - (repmat(r, [1,nPrj])-repmat(off, [1,nP])').^2);
  pr(pr<0) = 0;   
end

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