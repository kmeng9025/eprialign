% [fit_amp, fit_t1, fit_inv, fit_t2, fit_err_mask, fit_error, ff] = ...
%   fit_recovery_saturated(fit_y, taus, TT, TRs)
% fit to '(a - b*exp(-TT/T1))*exp(-taus/T2)' function
%   fit_y - data,  taus and TT - inversion recovery sequence times
%   res = ff([fit_amp, fit_t1, fit_inv, fit_t2], taus, TT)
%   fit_err_mask = index of good and bad fits, good - true, bad - false
%   fit_error = [error_amp, error_t1, error_inv, error_t2, residuals]

function [fit_amp, fit_t1, fit_inv, fit_t2, fit_err_mask, fit_error, ff] = ...
  fit_recovery_saturated(fit_y, taus, TT, TRs)

warning off;

% allocate arrays
sz = size(fit_y, 1);
fit_err_mask = zeros(1, sz);
fit_amp      = zeros(1, sz);
fit_t1       = zeros(1, sz);
fit_t2       = zeros(1, sz);
fit_inv      = zeros(1, sz);
fit_error    = zeros(5, sz);
x            = zeros(sz, 4);
fval         = zeros(sz, 1);

% condition the data add one more point for decay fit
fit_y = double(fit_y);
fit_y(:,end+1) = 0;
TT = [TT(:)',TT(end)]; 
taus = [taus(:)', 4*max(taus)];
TRs = [TRs(:)',TRs(end)]; 

% define fitting function
fit_recovery = @(x) (1 - 2*exp(-TRs/x(2)) + exp(-(TRs + taus/2)/x(2))).*(x(1) - x(3)*exp(-TT/x(2))).*exp(-taus/x(4));
ff = @(x,ataus,aTTs,aTRs) (1 - 2*exp(-aTRs/x(2)) + exp(-(aTRs + ataus/2)/x(2))).*(x(1) - x(3)*exp(-aTTs/x(2))).*exp(-ataus/x(4));

% estimation of an error
fit_exp_dx1 = @(x) exp(-taus/x(4)); % d(fit_exp)/dx(1)
fit_exp_dx2 = @(x)x(3)*exp(-TT/x(2)).*TT/x(2)/x(2).*exp(-taus/x(4)); % d(fit_exp)/dx(2)
fit_exp_dx3 = @(x)exp(-TT/x(2)).*exp(-taus/x(4)); % d(fit_exp)/dx(3)
fit_exp_dx4 = @(x)(x(1) - x(3)*exp(-TT/x(2))).*taus/x(4)/x(4).*exp(-taus/x(4)); % d(fit_exp)/dx(4)

% fit_exp_dx1 = @(x) (1 - x(2)*exp(-TRs/x(3)) + exp(-(TRs + taus/2)/x(3)))*exp(-taus/x(4)); % d(fit_exp)/dx(1)
% fit_exp_dx2 = @(x)  -x(1)*exp(-TRs/x(3))*exp(-taus/x(4)); % d(fit_exp)/dx(2)
% fit_exp_dx3 = @(x) (-x(1)*exp(-taus/x(4))*x(2))*(TRs/x(3)^2)*exp(-TRs/x(3)) + ...
%     exp(-taus/x(4))*((TRs + taus/2)/x(3)^2)*exp(-(TRs + taus/2)/x(3)); % d(fit_exp)/dx(3)
% fit_exp_dx4 = @(x) x(1)*(1 - x(2)*exp(-aTRs/x(3)) + exp(-(aTRs + ataus/2)/x(3)))*exp(-ataus/x(4))*(taus/x(4)^2); % d(fit_exp)/dx(4)
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
    [x(ii,:), fval(ii), fit_err_mask(ii)] = fminsearch(fit_fun, defaults(ii,:), []);
    
%     figure(5); clf; hold on;
%     plot(TT, fit_y(ii,:), '.');
%     plot(TT, fit_recovery(x(ii,:)), '-');
%     disp('!')
  end
  fit_amp = x(:, 1)'; fit_t1 = x(:, 2)'; fit_t2 = x(:, 4)'; fit_inv = x(:, 3)';
  fit_err_mask = fit_err_mask ~= -1;
  fit_error(5,:) = fval';
  
  for ii=1:sz
    residual_std = fval(ii)/sqrt(degree_of_freedom);
    J = [fit_exp_dx1(x(ii,:)); fit_exp_dx2(x(ii,:)); fit_exp_dx3(x(ii,:)); fit_exp_dx4(x(ii,:))]';
    Sigma = residual_std^2*inv(J'*J);
    se = sqrt(diag(Sigma))';
    fit_error(1:4,ii) = se;
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
