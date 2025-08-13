function [fit_amp, fit_t1, fit_inv,  fit_err_mask, fit_error, fit_recovery] = fit_recovery_3par_clean(fit_y, TT)

% Call original fit 
[fit_amp, fit_t1, fit_inv,  fit_err_mask, fit_error, fit_recovery] = fit_recovery_3par(fit_y, TT);

% Modify the fit
x = [fit_amp', (1./fit_t1)', fit_inv'];
y_true = fit_recovery(x, TT); % fit-recovery gets x and t as inputs
errors = real(y_true - fit_y);

this_is_it = 4*std(errors); % less than 0.1% of good points will be lost
outliers = abs(errors) > this_is_it;

if any(outliers)
  TT_new  = TT(~outliers);
  fit_y_new = fit_y(~outliers);
  disp('OUTLIERS!!!!!');
  
  %   figure(500); clf
  %   plot(TT, fit_y, 'o', TT, y_true, '-',  TT(outliers), fit_y(outliers), '*');
  
  % Call fit with modified data
  [fit_amp, fit_t1, fit_inv,  fit_err_mask, fit_error, fit_recovery] = fit_recovery_3par(fit_y_new, TT_new);
end


