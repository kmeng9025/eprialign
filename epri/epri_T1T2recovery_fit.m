function [fit_data, fit_data_info] = epri_T1T2recovery_fit(mat_recFXD, raw_info, fit_data_info)

TT   = raw_info.T1(:)'*1E6;
taus = 2*raw_info.tau(:)'*1E6;
TRs  = raw_info.Trep(:)'*1E6;


% Select mask
[mat_recMask, fit_data_info] = epri_GenerateFittingMask(mat_recFXD, fit_data_info);

% Fit decays
sz = size(mat_recFXD); number_of_echoes = sz(4);
sz1 = sz(1)*sz(2)*sz(3);
yyy = reshape(mat_recFXD,[sz1, number_of_echoes]);
idx = find(mat_recMask(:))';
fprintf('Fitting %i voxels.\n', numel(idx));

fit_y = double(yyy(idx,:));

recon_echosT1 = safeget(fit_data_info, 'use_echosT1', []);
recon_echosT2 = safeget(fit_data_info, 'use_echosT2', []);
recon_echosT1T2 = unique([recon_echosT1(:);recon_echosT2(:)]);
 
fit_data.Perr = zeros(5, size(fit_y,1));

switch safeget(fit_data_info, 'fit_function', 'fit_recovery_3par')
  case 'fit_recovery_3par'
    tauT1 = taus(1);
    if isempty(recon_echosT1) || isempty(recon_echosT2)
      idxT1 = taus == tauT1;
      idxT1(find(idxT1, 1, 'last')) = false;
      idxT2 = ~idxT1;
    else
      idxT1 = recon_echosT1;
      idxT2 = recon_echosT2;
    end
    switch safeget(fit_data_info, 'fit_method', 'default')
      case 'lookup_general'
        pars.func = @(x, A, B) (1 - 2*A*exp(-x*B));
        pars.par1 = eval(fit_data_info.fit_par_inv);
        pars.par2 = eval(fit_data_info.fit_par_R1);
        pars.x  =  TT(idxT1);
        
        % balance function (make inv close to 1)
        m1 = min(fit_y, [], 2);
        m2 = max(fit_y, [], 2);
        off = (m1+m2)/2;
        
        % make dictionary and fit
        res = fit_lookup2((fit_y(:,idxT1) - repmat(off, 1, 8))', pars, 2);
        
        fit_ampT1 = res(:,1)' + off';
        fit_t1  = 1./res(:,3)';
        fit_inv = (res(:,2).*res(:,1))'./fit_ampT1 ;
        
        fit_errorT1 = res(:,4)';
        
        
        pars1.func = @(x, A) exp(-x*A);
        pars1.par1 = eval(fit_data_info.fit_par_R2);
        pars1.x  =  taus(idxT2);
        
        % make dictionary and fit
        res = fit_lookup1(fit_y(:,idxT2)', pars1, 2);
        
%         fit_ampT2 = res(:,1)';
        fit_t2  = 1./res(:,2)';

        fit_amp = fit_ampT1./exp(-tauT1./fit_t2);
        fit_data.FitMask =  true(size(fit_ampT1));
        
      otherwise
        [fit_ampT2, fit_t2, fit_err_maskT2, fit_errorT2] = fit_exp_no_offset(fit_y(:,idxT2), taus(idxT2));
        [fit_ampT1, fit_t1, fit_inv, fit_err_maskT1, fit_errorT1] = fit_recovery_3par(fit_y(:,idxT1), TT(idxT1));
        fit_amp = fit_ampT1./exp(-tauT1./fit_t2);
        fit_data.FitMask = logical(fit_err_maskT1&fit_err_maskT2);
    end
  case 'fit_recovery_simultaneous'
    [fit_amp, fit_t1, fit_inv, fit_t2, fit_err_mask, fit_errorT1] = fit_recovery_simultaneous(fit_y(:,recon_echosT1T2), taus, TT);
    fit_data.FitMask = logical(fit_err_mask);
  case 'fit_recovery_saturated'
    [fit_amp, fit_t1, fit_inv, fit_t2, fit_err_mask, fit_errorT1] = fit_recovery_saturated(fit_y(:,recon_echosT1T2), taus, TT, TRs);
    fit_data.FitMask = logical(fit_err_mask);
end

fit_data.Algorithm = 'T2T1_InvRecovery_3Par';
fit_data.Size = sz(1:3);
fit_data.Parameters = {'Amplitude'; 'T1'; 'T2'; 'Inversion'};
fit_data.P    = [fit_amp; fit_t1; fit_t2; fit_inv];
%fit_data.Perr(5,:) = fit_errorT1; %%% Had to change to dodge a size error
fit_data.Perr = fit_errorT1;
fit_data.Idx  = idx;

% Remove errors
if strcmp(safeget(fit_data_info, 'fit_errors_kill','yes'),'yes')
  fit_data.Mask = fit_data.FitMask;
else
  fit_data.Mask = logical(fit_data.Idx);
end

% estimation of the error for T1
% f = A * (1 - 2*B*exp(-tt*C))
tt = TT(idxT1);
fit_exp_dx1 = @(B,C) 1 - 2*B*exp(-tt*C); % d(fit_exp)/dA
fit_exp_dx2 = @(A,C) 2*A*exp(-tt*C); % d(fit_exp)/dB
fit_exp_dx3 = @(A,B,C) -2*A*B*exp(-tt*C).*tt; % d(fit_exp)/dC
degree_of_freedom = length(tt) - 3;
fit_t1(fit_t1 < 0.1) = 0.1; % protection agains 0 division
r1 = 1./fit_t1;

%%% 8/16/17 MM had to comment OUT. Problem with Perr size ??%%%%
% warning off
% for ii=1:length(fit_amp)
%   rmse = fit_data.Perr(5,ii)/sqrt(degree_of_freedom);
%   J = [fit_exp_dx1(fit_inv(ii),r1(ii)); fit_exp_dx2(fit_ampT1(ii), r1(ii)); fit_exp_dx3(fit_ampT1(ii), fit_inv(ii), r1(ii))]';
%   Sigma = rmse^2*inv(J'*J);
%   se = sqrt(diag(Sigma))';
%   fit_data.Perr([1,4,2],ii) = se;
% end
% warning on

% Remove outlyers
avg = median(fit_data.P(1,fit_data.Mask)); 
fprintf('Median intensity = %g.\n', avg);
fit_data.Mask = fit_data.Mask & (fit_data.P(1,:) <= safeget(fit_data_info, 'fit_max_amp', 1E6)*avg);
fit_data.Mask = fit_data.Mask & (fit_data.P(1,:) >= safeget(fit_data_info, 'fit_min_amp', -1E6)*avg);
fit_data.Mask = fit_data.Mask & (fit_data.P(2,:) >= safeget(fit_data_info, 'fit_min_T1', 0.));
fit_data.Mask = fit_data.Mask & (fit_data.P(2,:) <= safeget(fit_data_info, 'fit_max_T1', 100.));

