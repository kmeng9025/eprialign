% function pO2 = epr_T2_PO2(T2, Amp, mask, pO2_info)
% convert T2 [us] to pO2 [torr] using constants from pO2_info structure

function pO2 = epr_T2star_PO2(T2, Amp, mask, pO2_info)

R2 = zeros(size(T2));
T2star0 = 0.75; % us
if ~exist('mask', 'var') || isempty(mask)
  R2 = 1./ T2 - 1/T2star0;
else
%   T2(T2 > T2star0) = T2star0;
  R2(mask) = 1./ T2(mask) - 1/T2star0;
end
R2star = R2/pi/2/2.802*1000; % in mG

LLW = R2star*0.75 + 25;

if ~exist('Amp', 'var')
  Amp = [];
end
if ~exist('mask', 'var')
  mask = true(size(T2));
end
if ~exist('pO2_info', 'var')
  pO2_info = [];
end

pO2 = epr_LLW_PO2(LLW, Amp, mask, pO2_info);