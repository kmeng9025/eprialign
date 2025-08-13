% function pO2 = epr_T2_PO2(T2, Amp, mask, pO2_info)
% convert T2 [us] to pO2 [torr] using constants from pO2_info structure

function pO2 = epr_T2_PO2(T2, Amp, mask, pO2_info)

R2 = zeros(size(T2));
if ~exist('mask', 'var') || isempty(mask)
  R2 = 1.0 ./ T2;
else
  R2(mask) = 1.0 ./ T2(mask);
end
LLW = R2/pi/2/2.802*1000; % in mG

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