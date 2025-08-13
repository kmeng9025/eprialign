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

function [fit_data, fit_data_info] = epri_findmax_fit(mat_recFXD, T1, fit_data_info)

% Select mask
[FitMask, fit_data_info] = epri_GenerateFittingMask(mat_recFXD, fit_data_info);

% Fit decays
sz = size(mat_recFXD); number_of_echoes = sz(4);
sz1 = sz(1)*sz(2)*sz(3);
yyy = reshape(mat_recFXD,[sz1, number_of_echoes]);
idx = find(FitMask(:))';
fprintf('Fitting %i voxels.\n', numel(idx));

tic
fit_y = double(yyy(idx,:));
fit_data.Perr = zeros(1, size(fit_y,1));
[A,B] = max(fit_y, [], 2);

% R1 fit header
fit_data.Algorithm = 'max_1Par';
fit_data.Size = sz(1:3);
fit_data.Parameters = {'Amplitude'; 'Index'};
fit_data.P    = [A, B]';
fit_data.Idx  = idx;
fit_data.FitMask = true(size(fit_data.Idx));
toc

