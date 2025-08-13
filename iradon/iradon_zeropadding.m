% IRADON_ZEROPADDING   zeropadding of the data first dimension
% function [Pzp, zeropadding] = iradon_zeropadding(P, zeropadding)
% P           - Data, [array, any dimensions]
% zeropadding - Zeropadding factor [int, >= 1]
% Pzp         - Output data [array, any dimensions]


% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago,JULY 2013
% Contact: epri.uchicago.edu

function [Pzp, zeropadding] = iradon_zeropadding(P, zeropadding)

np = size(P);
zeropadding = max(1, zeropadding);

if zeropadding > 1
  nPad = fix(np(1)*(zeropadding-1)/2);
  nzp = np; nzp(1) = nPad;
  zp = zeros(nzp);
  Pzp = [zp; P; zp];
  zeropadding = (nPad*2+np(1)) / np(1);
else
  Pzp = P;
end
