% [fit_amp, fit_t1, fit_inv, fit_t2, fit_err_mask, fit_error, ff] = ...
%   fit_recovery_simultaneous(fit_y, tt, TT)
% fit to '(a - b*exp(-TT/T1))*exp(-tt/T2)' function
%   fit_y - data,  tt and TT - inversion recovery sequence times
%   res = ff([fit_amp, fit_t1, fit_inv, fit_t2], tt, TT)
%   fit_err_mask = index of good and bad fits, good - true, bad - false
%   fit_error = [error_amp, error_t1, error_inv, error_t2, residuals]

% function [fit_amp, fit_t1, fit_inv, fit_t2, fit_err_mask, fit_error, ff] = ...
%   fit_recovery_simultaneous(fit_y, tt, TT)
function [fitted] = fit_recovery_simultaneous_GR(fit_y, tt, TT)

warning off;

% allocate arrays
sz = size(fit_y, 1);
fitted.err_mask = zeros(1, sz);
fitted.amp      = zeros(1, sz);
fitted.T1       = zeros(1, sz);
fitted.inv      = zeros(1, sz);
fitted.error    = zeros(5, sz);
x            = zeros(sz, 4);
fval         = zeros(sz, 1);

% condition the data add one more point for decay fit
fit_y = double(fit_y);
fit_y(:,end+1) = 0;
TT = [TT(:)',TT(end)]; 
tt = [tt(:)', 4*max(tt)];

% define fitting function
fit_recovery = @(x) (x(1) - x(3)*exp(-TT/x(2))).*exp(-tt/x(4));
fitted.ff = @(x,att,aTTs) (x(1) - x(3)*exp(-aTTs/x(2))).*exp(-att/x(4));

% estimation of an error
fit_exp_dx1 = @(x) exp(-tt/x(4)); % d(fit_exp)/dx(1)
fit_exp_dx2 = @(x)x(3)*exp(-TT/x(2)).*TT/x(2)/x(2).*exp(-tt/x(4)); % d(fit_exp)/dx(2)
fit_exp_dx3 = @(x)exp(-TT/x(2)).*exp(-tt/x(4)); % d(fit_exp)/dx(3)
fit_exp_dx4 = @(x)(x(1) - x(3)*exp(-TT/x(2))).*tt/x(4)/x(4).*exp(-tt/x(4)); % d(fit_exp)/dx(4)
degree_of_freedom = length(TT) - 2;

p = sum(fit_y, 1);
fit_fun = @(x) sqrt(sum((p - fit_recovery(x)).^2));
est_TT = fminsearch(fit_fun, [max(p),max(TT)/4,max(p),max(TT)/4], []);

defaults = [max(fit_y,[],2),repmat(est_TT(2), sz, 1),max(fit_y,[],2),repmat(est_TT(4), sz, 1)];

tic
set(gcf,'Pointer','watch');drawnow
try
  %   optionsdef = optimset('lsqcurvefit');
  %   optionsnew = optimset('Display', 'off', 'diagnostics', 'off');
  %   options = optimset(optionsdef, optionsnew);
  
  parfor ii=1:sz
    fit_fun = @(x) sqrt(sum((fit_y(ii,:) - fit_recovery(x)).^2));
    [x(ii,:), fval(ii), fitted.err_mask(ii)] = fminsearch(fit_fun, defaults(ii,:), []);
    
%     figure(5); clf; hold on;
%     plot(TT, fit_y(ii,:), '.');
%     plot(TT, fit_recovery(x(ii,:)), '-');
%     disp('!')
  end
  fitted.amp = x(:, 1)'; fitted.T1 = x(:, 2)'; fitted.T2 = x(:, 4)'; fitted.inv = x(:, 3)';
  fitted.err_mask = fitted.err_mask ~= -1;
  fitted.error(5,:) = fval';
  
  for ii=1:sz
    residual_std = fval(ii)/sqrt(degree_of_freedom);
    J = [fit_exp_dx1(x(ii,:)); fit_exp_dx2(x(ii,:)); fit_exp_dx3(x(ii,:)); fit_exp_dx4(x(ii,:))]';
    Sigma = residual_std^2*inv(J'*J);
    se = sqrt(diag(Sigma))';
    fitted.error(1:4,ii) = se;
  end
  
catch err
  disp(sprintf('PulseFitGUI: %s.', err.message))
end
warning on
set(gcf,'Pointer','arrow');drawnow

if sz(1) > 10
  toc
  disp('Fitting is finished.');
end
