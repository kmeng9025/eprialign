function [ m4d ] = radon_addspectrum( M, spectr )
%Radon_addspectrum 
%   Extends 3D phantom into 4D along a defined spectrum
%   M - matrix of an object (float, NxNxN - matrix dimension)
%   spectr - spectrum to be added
%   
%   for type Lorentz
%       - center - center of the distribution
%       - sigma - width

msize=size(M);

nBins4   = safeget(spectr, 'nBins4', 64);
matSizeG = safeget(spectr, 'matSizeG', 1.024*sqrt(2));
x = linspace(-matSizeG/2,matSizeG/2,nBins4);

sigma = safeget(spectr, 'lw', 5);
center = safeget(spectr, 'center', 0);
phase  = safeget(spectr, 'phase', 0) * pi/180;

switch upper(safeget(spectr, 'spectrum', 'LORENTZIAN'))
    case 'LORENTZIAN'
        spectrum = real(exp(-1i*phase)*epr_Lorentzian(x, center, sigma(1)));
    otherwise
end

Mall = M(:);
idx = find(Mall > 0);
m4d=zeros([numel(M), msize(1)]);

for ii = 1:size(idx)
  m4d(idx(ii),:)=spectrum * Mall(idx(ii));
end

m4d = reshape(m4d, [msize, msize(1)]);
