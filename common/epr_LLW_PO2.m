% function pO2 = epr_LW_PO2(LLW, Amp, mask, pO2_info)
% convert LLW [mGauss] to pO2 [torr] using constants from pO2_info structure

function pO2 = epr_LLW_PO2(LLW, Amp, mask, pO2_info)

if ~exist('Amp', 'var')
  Amp = [];
end
if ~exist('pO2_info', 'var')
  pO2_info = [];
end

LLW_zero_po2    = safeget(pO2_info, 'LLW_zero_po2', 10.2);
Torr_per_mGauss = safeget(pO2_info, 'Torr_per_mGauss', 1.84);
mG_per_mM       = safeget(pO2_info, 'mG_per_mM', 0);
MDNmG_per_mM    = safeget(pO2_info, 'MDNmG_per_mM', 0);

if isempty(pO2_info)
  fprintf('LLW_zero_po2: %4.2f mG\n' ,LLW_zero_po2);
  fprintf('Torr_per_mGauss: %4.2f torr/mG\n' ,Torr_per_mGauss);
  fprintf('mG_per_mM: %4.2f mG/mM\n' ,mG_per_mM);
  fprintf('MDNmG_per_mM: %4.2f mG/mM\n' ,MDNmG_per_mM);
end

pO2 = zeros(size(LLW));
if ~exist('mask', 'var') || isempty(mask)
    pO2 = (LLW - LLW_zero_po2) * Torr_per_mGauss;
elseif isempty(Amp)
  pO2(mask) = (LLW(mask) - LLW_zero_po2) * Torr_per_mGauss;
else
  amp1mM       = safeget(pO2_info, 'amp1mM', mean(Amp(mask(:))));
  ampAvg       = median(Amp(mask(:)))/amp1mM;
  pO2(mask) = (LLW(mask) - LLW_zero_po2 - Amp(mask)/amp1mM*mG_per_mM - ampAvg*MDNmG_per_mM) * Torr_per_mGauss;
end
