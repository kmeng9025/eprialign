% EPRI_DECAY_FIT - lookup dictionary fitting to a function with 1 parameters
% [fit_data, fit_data_info] = EPRI_DECAY_FIT(mat_recFXD, decay_tau2, fit_data_info)
% mat_recFXD - input 4D image, 3D NxNxN points, 4D trace M points [double NxNxNxM]
% decay_tau2 - 2x'tau' [double 1xM]
% fit_data_info  - [structure] of fit parameters   
%   [].xxxx - Function of 3 arguments [f.e. @(x, A) exp(-x/A)]
% fit_data  - Generate dictionary/fit data using dictionary/both [0 | 1 | 2]
% See also EPRI_RECOVERY_FIT

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago,JULY 2013
% Contact: epri.uchicago.edu

function [fit_data, fit_data_info] = epri_decay_fit(mat_recFXD, decay_tau2, fit_data_info)

% Select mask
[mat_recMask, fit_data_info] = epri_GenerateFittingMask(mat_recFXD, fit_data_info);

% Fit decays
sz = size(mat_recFXD); number_of_echoes = sz(4);
sz1 = sz(1)*sz(2)*sz(3);
yyy = reshape(mat_recFXD,[sz1, number_of_echoes]);
idx = find(mat_recMask(:))';
fprintf(sprintf('Fitting %i voxels.\n', numel(idx)));

% tau = mat_info.T1(:)' * 1E6; % in us
fit_y = double(yyy(idx,:));

% add one more point at very long tau to fix the offset
decay_tau2 = [decay_tau2(:); max(decay_tau2)*4]';
fit_y(:,end+1) = 0;
fit_data.Perr = zeros(3, size(fit_y,1));

tic
switch safeget(fit_data_info, 'fit_method', 'default')
  case 'lookup_general'
    pars.func = @(x, A) exp(-x*A);
    pars.par1 = eval(fit_data_info.fit_par_R2);
    pars.x  =  decay_tau2(:);

    % make dictionary and fit
    res = fit_lookup1(fit_y', pars, 2);
    
    fit_amp = res(:,1)';
    fit_t2  = 1./res(:,2)';
    
    % Error to be calculated
    fit_err_mask  = true(size(fit_amp));
    fit_data.Perr(3,:) = res(:,3)';
  otherwise
    [fit_amp, fit_t2, fit_err_mask, fit_error] = fit_exp_no_offset(fit_y, decay_tau2);
    fit_data.Perr(3,:) = fit_error;
end

fit_data.Algorithm = 'T2_ExpDecay_No_Offset';
fit_data.Size = sz(1:3);
fit_data.Parameters = {'Amplitude'; 'T2'};
fit_data.P    = [fit_amp; fit_t2];
fit_data.Idx  = idx;
fit_data.FitMask = logical(fit_err_mask);

% Fit error estimation
% f = A*exp(-tt*B)
tt = decay_tau2(:)';
fit_exp_dx1 = @(B)exp(-tt*B); % d(fit_exp)/dA
fit_exp_dx2 = @(A,B)-A*exp(-tt*B).*tt; % d(fit_exp)/dB
degree_of_freedom = length(tt) - 2;
fit_t2(fit_t2 < 0.1) = 0.1; % protection agains 0 division
r2 = 1./fit_t2;

% for ii=1:length(fit_amp)
%   rmse = fit_data.Perr(3,ii)/sqrt(degree_of_freedom);
%   J = [fit_exp_dx1(r2(ii)); fit_exp_dx2(fit_amp(ii), r2(ii))]';
%   se = rmse*sqrt(diag(inv(J'*J)))';
%   fit_data.Perr(1:2,ii) = se;
% end
toc

% Remove errors
if strcmp(safeget(fit_data_info, 'fit_errors_kill','yes'),'yes')
  fit_data.Mask = fit_data.FitMask;
else
  fit_data.Mask = logical(fit_data.Idx);
end

% Remove outlyers
avg = median(fit_data.P(1,fit_data.Mask)); 
fprintf('Median intensity = %g.\n', avg);
fit_data.Mask = fit_data.Mask & (fit_data.P(1,:) <= safeget(fit_data_info, 'fit_max_amp', 1E6)*avg);
fit_data.Mask = fit_data.Mask & (fit_data.P(1,:) >= safeget(fit_data_info, 'fit_min_amp', -1E6)*avg);
fit_data.Mask = fit_data.Mask & (fit_data.P(2,:) >= safeget(fit_data_info, 'fit_min_T2', 0.));
fit_data.Mask = fit_data.Mask & (fit_data.P(2,:) <= safeget(fit_data_info, 'fit_max_T2', 100.));

% --------------------------------------------------------------------
% function saturation_correction = T1_correction(tau, Treps, T1)
% 
% saturation_correction = 1./(1 - 2*exp(-(Treps-tau)./T1) + exp(-Treps./T1));
