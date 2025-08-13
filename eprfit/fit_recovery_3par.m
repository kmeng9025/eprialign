% [fit_amp, fit_t1, fit_inv, fit_err_mask, fit_error, ff] = fit_recovery_3par(fit_y, TT)
% fit to 'a - b*exp(-TT/T1)' function
%   fit_y - data,  taus and TT - inversion recovery sequence times
%   res = ff([fit_amp, fit_t1, fit_inv], TT)
%   fit_err_mask = index of good and bad fits, good - true, bad - false
%   fit_error = [error_amp, error_t1, error_inv, residuals]

function [fit_amp, fit_t1, fit_inv, fit_err_mask, fit_error, fit_recovery] = fit_recovery_3par(fit_y, TT)

warning off;

% allocate arrays
sz = size(fit_y, 1);
fit_err_mask = zeros(1, sz);
fit_amp      = zeros(1, sz);
fit_t1       = zeros(1, sz);
fit_inv      = zeros(1, sz);
fit_error    = zeros(4, sz);
x            = zeros(sz, 3);
fval         = zeros(sz, 1);

% define fitting function
fit_recovery = @(x,t) x(1)*(1 - 2*x(3)*exp(-t*x(2)));

if isempty(fit_y), return; end

% condition the data
fit_y = double(fit_y);
TT = TT(:)';
Trange = max(TT);

p = sum(fit_y, 1);
fit_fun = @(x) sqrt(sum((p - fit_recovery(x,TT)).^2));
opt = optimset('Display','off');
[est_TT, err] = fminsearch(fit_fun, [max(p),1/(Trange/5),max(p)], opt);

defaults = [max(fit_y,[],2),est_TT(2)*ones(sz, 1),ones(sz, 1)];

tic
try
  %   optionsdef = optimset('lsqcurvefit');
  %   optionsnew = optimset('Display', 'off', 'diagnostics', 'off');
  %   options = optimset(optionsdef, optionsnew);
  
  if sz > 100
      parfor ii=1:sz
          fit_fun = @(x) sqrt(sum((fit_y(ii,:) - fit_recovery(x,TT)).^2));
          [x(ii,:), fval(ii), fit_err_mask(ii)] = fminsearch(fit_fun, defaults(ii,:), opt);
          %       figure(5); clf; hold on;
          %       plot(TT, fit_y(ii,:), '.');
          %       plot(TT, fit_recovery(x(ii,:),TT), '-');
          %       disp('!')
      end
  else
      for ii=1:sz
          fit_fun = @(x) sqrt(sum((fit_y(ii,:) - fit_recovery(x,TT)).^2));
          [x(ii,:), fval(ii), fit_err_mask(ii)] = fminsearch(fit_fun, defaults(ii,:), opt);
      end
  end
  fit_amp = x(:, 1)'; fit_t1 = 1./(x(:, 2)'); fit_inv = x(:, 3)';
  fit_err_mask = fit_err_mask ~= -1;
%   fit_error = fval';
    
  % error estimation
%   https://www.mathworks.com/matlabcentral/newsreader/view_thread/157530
  recovery_dx1 = @(x) (1 - 2*x(3)*exp(-TT*x(2)));
  recovery_dx2 = @(x) 2*TT*x(1)*x(3).*exp(-TT*x(2));
  recovery_dx3 = @(x) -x(1)*2*exp(-TT*x(2));
  degree_of_freedom = length(TT) - 3;
  for ii=1:sz
    residual_std = fval(ii)/sqrt(degree_of_freedom);
    J = [recovery_dx1(x(ii,:)); recovery_dx2(x(ii,:)); recovery_dx3(x(ii,:))]';
    Sigma = residual_std^2*inv(J'*J);
    se = sqrt(diag(Sigma));
    fit_error(1:3,ii) = se;
    fit_error(4,ii) = fval(ii);
  end

catch err
  fprintf('PulseFitGUI: %s.\n', err.message);
end
warning on

if sz(1) > 10
  toc
  disp('Fitting is finished.');
end
