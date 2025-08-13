function [mat_fit, mat_fit_info] = rs_spectral_fit(mat_recFXD, raw_info, mat_fit_info)

% Select mask
mat_fit_info.data = 'absorption_line';
[mat_recMask, mat_fit_info] = epri_GenerateFittingMask(mat_recFXD, mat_fit_info);

Cshift = safeget(raw_info, 'Boffset', 0);
% Fit
tic
sz = size(mat_recFXD); 
sz1 = sz(1)*sz(2)*sz(3);
yyy = reshape(mat_recFXD,[sz1, sz(4)]);
idx = find(mat_recMask(:))';
disp(sprintf('Fitting %i voxels.', numel(idx)));

data = yyy(idx, :);

scan_width=raw_info.deltaH;
B = (1:sz(4))' * scan_width / (sz(4)- 0); B = B - mean(B);

mat_fit_info.spin_probe = safeget(mat_fit_info, 'spin_probe', 'OX063H');

[mat_fit.P, mat_fit.Perr] = fit_rs_amp_R2_phase_xover(B, data', mat_fit_info);

mat_fit.P = mat_fit.P + Cshift;
fprintf('rs_spectral_fit: XOVER was corrected by %4.3f G\n', Cshift);

mat_fit.Algorithm = 'CW_spectral_fit_R2_XOVR_PHASE';
mat_fit.Size = sz(1:3);
mat_fit.Parameters = {'Amplitude'; 'R2'; 'X-over'; 'Phase'; 'Error'};
mat_fit.Idx  = idx;
toc
% mat_fit.FitMask = logical(fit_err_mask);

% Remove errors
% if strcmp(safeget(mat_fit_info, 'fit_errors_kill','yes'),'yes')
%   mat_fit.Mask = mat_fit.FitMask;
% else
%   mat_fit.Mask = logical(mat_fit.Idx);
% end

% Remove outliers
% avg = mean(mat_fit.P(1,mat_fit.Mask)); disp(sprintf('Average intensity = %g', avg));
% mat_fit.Mask = mat_fit.Mask & (mat_fit.P(1,:) <= safeget(mat_fit_info, 'fit_max_amp')*avg);
% avg = mean(mat_fit.P(1,mat_fit.Mask));
% mat_fit.Mask = mat_fit.Mask & (mat_fit.P(1,:) >= safeget(mat_fit_info, 'fit_min_amp')*avg);
% mat_fit.Mask = mat_fit.Mask & (mat_fit.P(2,:) >= safeget(mat_fit_info, 'fit_min_LLW'));
% mat_fit.Mask = mat_fit.Mask & (mat_fit.P(2,:) <= safeget(mat_fit_info, 'fit_max_LLW'));

