% EPRI_RECOVERY_FIT - lookup dictionary fitting to a function with 1 parameters
% [fit_data, fit_data_info] = EPRI_RECOVERY_FIT(mat_recFXD, decay_tau2, fit_data_info)
% mat_recFXD - input 4D image, 3D NxNxN points, 4D trace M points [double NxNxNxM]
% decay_tau2 - 2x'tau' [double 1xM]
% fit_data_info  - [structure] of fit parameters   
%   [].func - Function of 3 arguments [f.e. @(x, A) exp(-x/A)]
%   [].par1 - Array of first argument values [double array]
%   [].x    - Array of function arguments [double array, Nx1]
%   [].ndict - Normalized dictionary array
%   [].normV - Norm of dictionary
% fit_data  - Generate dictionary/fit data using dictionary/both [0 | 1 | 2]
% See also EPRI_DECAY_FIT

function [fit_data, fit_data_info] = epri_recovery_fit(mat_recFXD, T1, fit_data_info)

% Select mask
[FitMask, fit_data_info] = epri_GenerateFittingMask(mat_recFXD, fit_data_info);

% fit_recovery_flag = contains(safeget(fit_data_info, 'fit_recovery_range', 'no'), 'yes');
% % 
% %Fit decays
% if fit_recovery_flag
%     T1 = [T1, 1.5*T1(end)];
%     mat_recFXD(:,:,:,11) = mat_recFXD(:,:,:,10);
% end

sz = size(mat_recFXD); number_of_echoes = sz(4);
sz1 = sz(1)*sz(2)*sz(3);
yyy = reshape(mat_recFXD,[sz1, number_of_echoes]);
idx = find(FitMask(:))';
fprintf('Fitting %i voxels.\n', numel(idx));

tic
fit_y = double(yyy(idx,:));
fit_data.Perr = zeros(4, size(fit_y,1));
switch safeget(fit_data_info, 'fit_method', 'default')
  case 'lookup_general'
    pars.func = @(x, A, B) (1 - 2*A*exp(-x*B));
    pars.par1 = eval(fit_data_info.fit_par_inv);
    pars.par2 = eval(fit_data_info.fit_par_R1);
    pars.x  =  T1(:);
    nT1 = length(T1);

    % balance function (make inv close to 1)
    m1 = min(fit_y, [], 2);
    m2 = max(fit_y, [], 2);
    off = (m1+m2)/2;
    
    % make dictionary and fit
    res = fit_lookup2((fit_y - repmat(off, 1, nT1))', pars, 2);
    
    fit_amp = res(:,1)' + off';
    fit_t1  = 1./res(:,3)';
    fit_inv = (res(:,2).*res(:,1))'./fit_amp ;
    
    % Error to be calculated
    fit_err_mask  = true(size(fit_amp));
    fit_data.Perr(4,:) = res(:,4)';
  case 'lookup_adapted'
    res = fit_lookuptable_T1(fit_y, 0.75*zeros(1,8), T1);
    fit_amp = res.amp;
    fit_t1  = res.T1;
    fit_inv = res.inv;
    fit_err_mask  = true(size(fit_amp));
    fit_data.Perr(4,:) = res(:,4)';
  otherwise
% use R1 fit function instead of t1 fit function    
%     [fit_amp, fit_t1, fit_inv, fit_err_mask, fit_data.Perr] = ...
%       fit_recovery_3par(fit_y, T1);
    [fit_amp, fit_r1, fit_inv, fit_err_mask, fit_data.Perr] = ...
      fit_recovery_3parR1(fit_y, T1);
end

% fit_data.Algorithm = 'T1_InvRecovery_3Par';
% fit_data.Size = sz(1:3);
% fit_data.Parameters = {'Amplitude'; 'T1'; 'Inversion'};
% fit_data.P    = [fit_amp; fit_t1; fit_inv];
% fit_data.Idx  = idx;
% fit_data.FitMask = logical(fit_err_mask);

% R1 fit header
fit_data.Algorithm = 'T1_InvRecovery_3ParR1';
fit_data.Size = sz(1:3);
fit_data.Parameters = {'Amplitude'; 'R1'; 'Inversion'};
fit_data.P    = [fit_amp; fit_r1; fit_inv];
fit_data.Idx  = idx;
fit_data.FitMask = logical(fit_err_mask);

toc

% Remove errors and outlyers
if strcmp(safeget(fit_data_info, 'fit_errors_kill','yes'),'yes')
    fit_data.Mask = fit_data.FitMask;

    % Remove outlyers
    avg = median(fit_data.P(1,fit_data.Mask));
    % Errors are still specified in T1 units
    % T1matrix = fit_data.P(2,:);
    T1matrix = 1./fit_data.P(2,:);

    fprintf('Median intensity = %g.\n', avg);
    fit_data.Mask = fit_data.Mask & (fit_data.P(1,:) <= safeget(fit_data_info, 'fit_max_amp', 1E6)*avg);
    fit_data.Mask = fit_data.Mask & (fit_data.P(1,:) >= safeget(fit_data_info, 'fit_min_amp', -1E6)*avg);
    fit_data.Mask = fit_data.Mask & (T1matrix >= safeget(fit_data_info, 'fit_min_T1', 0.));
    fit_data.Mask = fit_data.Mask & (T1matrix <= safeget(fit_data_info, 'fit_max_T1', 100.));

else
    fit_data.Mask = logical(fit_data.Idx);
end

