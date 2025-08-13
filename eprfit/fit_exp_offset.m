function [fit_amp, fit_t2, fit_off, fit_err_mask, fit_error, fit_exp] = fit_exp_offset(fit_y, tt)

% define fitting function
fit_exp = @(x,t) x(1)*exp(-t*x(2)) + x(3);

% allocate arrays
sz = size(fit_y, 1);
fit_err_mask = zeros(1, sz);
fit_amp      = zeros(1, sz);
fit_t2       = zeros(1, sz);
fit_off       = zeros(1, sz);
fit_error    = zeros(1, sz);
x            = zeros(sz, 3);
fval         = zeros(sz, 1);

if isempty(fit_y), return; end

warning off;

% condition the data
fit_y = double(fit_y);
tt = tt(:)';

%first estimate
p = sum(fit_y, 1)/sz;
fit_fun = @(x) sqrt(sum((p - fit_exp(x, tt)).^2));
x_estimate = [max(p),1/(max(tt(1:end-1))/4),0];
x_estimate2 = fminsearch(fit_fun, x_estimate, []);

% figure(10); clf
% plot(tt, fit_y, 'o', tt, fit_exp(x_estimate2, tt));
% legend({'data', 'fit'});

defaults = repmat(x_estimate2, [sz,1]);
opt = optimset('Display','off');

set(gcf,'Pointer','watch');drawnow
try
  %   optionsdef = optimset('lsqcurvefit');
  %   optionsnew = optimset('Display', 'off', 'diagnostics', 'off');
  %   options = optimset(optionsdef, optionsnew);
  if sz > 10
    parfor ii=1:sz
      fit_fun = @(x) sqrt(sum((fit_y(ii,:) - fit_exp(x,tt)).^2));
      [x(ii,:), fval(ii), fit_err_mask(ii)] = fminsearch(fit_fun, defaults(ii,:), opt);
    end
  else
    for ii=1:sz
      fit_fun = @(x) sqrt(sum((fit_y(ii,:) - fit_exp(x,tt)).^2));
      [x(ii,:), fval(ii), fit_err_mask(ii)] = fminsearch(fit_fun, defaults(ii,:), opt);
    end
  end
  fit_amp = x(:, 1)'; fit_t2 = 1./(x(:, 2)'); fit_off=x(:,3)';
  fit_err_mask = fit_err_mask ~= -1;
  fit_error = fval';
  
 
  % C Mailer grad search algorithm
%     pars.idx = 1; pars.PG = defaults(2); pars.yy = fit_y(ii,:)'; pars.xx = tt';
%     [x_cw(2), xx_err]=cw_grad_min(fit_exp_cw, pars);
%     fit_yy = fit_exp_cw(x_cw(2), pars);
%     x_cw(1) = (pars.yy' * fit_yy) / (fit_yy' * fit_yy);
%     x = x_cw; eflag = 1; fval = xx_err;
    
%     [x, fval, ff, eflag] = lsqcurvefit(fit_exp,x,tt,fit_y(ii,:),[0,1], [1,20], ...
%       options);
    
%     residual_std = fval/sqrt(degree_of_freedom);
%     J = [fit_exp_dx1(x); fit_exp_dx2(x)]';
%     Sigma = residual_std^2*inv(J'*J);
%     se = sqrt(diag(Sigma))';
% 
%     fit_error(1:2,ii) = se;
%     fit_error(3,ii) = fval;
%     fit_amp(ii) = x(1); fit_t2(ii)  = x(2);
%     fit_err_mask(ii) = eflag == 1;

%     if fit_amp(ii) > 0.2
%       figure(100); clf
%       plot(tt(1:end-1), fit_y(ii,1:end-1),'o'); hold on
%       xx = linspace(tt(1), tt(end-1), 20);
%       plot(xx, fit_exp(x,xx), xx, fit_exp(x_cw,xx));
%       text(.5, .85, sprintf('amp=%g, t2=%gus', fit_amp(ii), fit_t2(ii)), 'Unit', 'normalized')
%       xlabel('tt * 2 [us]'); axis tight
%       pause
%     end
%   end
catch err
  disp(sprintf('fit_exp_offset: %s.', err.message))
end
warning on
set(gcf,'Pointer','arrow');drawnow

if sz(1) > 10
  disp('Fitting is finished.');
end