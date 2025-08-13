% function pO2 = epr_T2_PO2(T2, Amp, mask, pO2_info)
% convert T2 [us] to pO2 [torr] using constants from pO2_info structure

function pO2 = epr_T2_PO2(T2, Amp, mask, pO2_info)

R2 = zeros(size(T2));

R2(mask) = 1.0 ./ T2(mask);
LLW = R2/pi/2/2.8*1000; % in mG

pO2 = epr_LLW_PO2(LLW, Amp, mask, pO2_info);