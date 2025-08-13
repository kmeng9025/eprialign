% function pO2 = epr_T2_PO2(T2, Amp, mask, pO2_info)
% convert T2 [us] to pO2 [torr] using constants from pO2_info structure

function pO2 = epr_R2_PO2(R2, Amp, mask, pO2_info)

LLW = R2/pi/2/2.8*1000; % in mG
switch nargin
  case 1
    pO2 = epr_LLW_PO2(LLW);
  otherwise
    pO2 = epr_LLW_PO2(LLW, Amp, mask, pO2_info);
end