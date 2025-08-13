function [isocenter,results] = TargetHypoxicSphere2(po2image, resolution, Mask, R)

% R = 2.5;
% resolution = [0.66,0.66,0.66];

% choice = questdlg('Which mask has been supplied?','Mask Type','Tumor Mask','EPR Mask','Cancel','Tumor Mask');
% switch choice
%     case 'Tumor Mask'
%         
tic

hypoxic_vox =  po2image <= 10 & po2image > -25;
fitted_vox =  po2image > -25;
normox_vox = po2image > 10;

tumormask = Mask&fitted_vox;
nottumormask = (~tumormask) & fitted_vox;

voxel_volume = prod(resolution);
nSize = size(po2image);
[X, Y, Z] = meshgrid(resolution(1)*(1:nSize(1)), resolution(2)*(1:nSize(2)), resolution(3)*(1:nSize(3)));

idxT = find(tumormask);
nTumor = length(idxT);
XT  = X(idxT);
YT  = Y(idxT);
ZT  = Z(idxT);

n_tumor_hypoxicvox =  numel(find(tumormask & hypoxic_vox));
n_tumor_normoxvox = numel(find(tumormask & normox_vox));

hypoxic_tumor_insphere = zeros(nTumor, 1);
normtiss_insphere = zeros(nTumor, 1);
normoxic_tumor_insphere = zeros(nTumor, 1);
tumor_outsphere = zeros(nTumor, 1);

for ii=1:nTumor
  radiation_mask = ((X-XT(ii)).^2 + (Y-YT(ii)).^2 + (Z-ZT(ii)).^2) < R^2;
  hypoxic_tumor_insphere(ii) = numel(find(radiation_mask & tumormask & hypoxic_vox));
  normtiss_insphere(ii) = numel(find(radiation_mask & nottumormask));
  normoxic_tumor_insphere(ii) = numel(find(radiation_mask & tumormask & normox_vox));
  tumor_outsphere(ii) = numel(find(~radiation_mask & tumormask));
end

[~,maxidx] = max(hypoxic_tumor_insphere);
if length(maxidx) > 1
  temp = normtiss_insphere(maxidx);
  [~,minidx] = min(temp);
  bestidx = maxidx(minidx);
else
  bestidx = maxidx;
end

results.hit_volume = 4/3*pi*R^3;
results.hypoxic_tumor_insphere_fraction = hypoxic_tumor_insphere(bestidx) / n_tumor_hypoxicvox;
results.hypoxic_tumor_insphere_volume = hypoxic_tumor_insphere(bestidx) * voxel_volume;
results.normtiss_insphere_volume = normtiss_insphere(bestidx) * voxel_volume;
results.normoxic_tumor_insphere_fraction = normoxic_tumor_insphere(bestidx) / n_tumor_normoxvox;
results.normoxic_tumor_insphere_volume = normoxic_tumor_insphere(bestidx)*voxel_volume;
results.tumor_outsphere_fraction = tumor_outsphere(bestidx) / nTumor;
results.tumor_insphere_fraction = 1-results.tumor_outsphere_fraction;

isocenter = [YT(bestidx),XT(bestidx),ZT(bestidx)] ./ resolution;
results.isocenter = isocenter;
toc
  
% end