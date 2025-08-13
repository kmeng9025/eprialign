function [ph, x_0, lw] = rs_get_phase(x,y,fit_pars)

Npt = length(y);
x = x(:); y = y(:);

% linear baseline correction
use_baseline = min(100, Npt/20);
bl_idx = [1:use_baseline,Npt-use_baseline:Npt];
pp = polyfit(x(bl_idx), y(bl_idx), 1);
y = y -  polyval(pp,x);

% ignore the outer parts of the spectrum
LSum = sum(abs(y)*mean(diff(x)));
estim_y = abs(y);
[LMax,b] = max(estim_y);
Fmax = x(b);
G = (max(x(estim_y > LMax / 2)) - min(x(estim_y > LMax / 2)))/2;

idx = (x > (Fmax-4*G)) & (x < (Fmax+4*G));
idxL = (x > (Fmax-4*G)) & (x < Fmax);

fit_B = x(idx); fit_B = fit_B(:);
yy = y(idx);

kphase = 1;
% adjust channels to correct symmetry
if sum(imag(y(idxL))) < 0 
  yy = real(yy)-1i*imag(yy);
end

% figure(100); clf;
% plot(fit_B, real(yy), fit_B, imag(yy))

% Select model
spin_probe = safeget(fit_pars, 'spin_probe', 'OX063D24');

if strcmpi(spin_probe, 'LORENTZIAN')
  fit_f = @(xx) xx(1)*exp(-1i*xx(4))*epr_Lorentzian(fit_B, xx(2), abs(xx(3)));
  fit_f_abs = @(xx) abs(xx(1)*epr_Lorentzian(fit_B, xx(2), abs(xx(3))));
  fmin = @(xx) norm(yy - fit_f(xx));
  fmin_abs = @(xx) norm(abs(yy) - fit_f_abs(xx));
   
  % estimation of starting conditions
  x1 = [LSum/pi/G/2, Fmax, G];
  x4 = fminsearch(fmin_abs, x1, []);
  x4([4]) = 0;
  x2 = fminsearch(fmin, x4);
  
%   % test function
%   figure(100); clf;
% %   test_f = fit_f([1, Fmax, G, 0], fit_B);
%   test_f = fit_f(x4);
%   plot(fit_B, real(yy), fit_B, imag(yy), ...
%     fit_B, real(test_f), fit_B, imag(test_f), ...
%     fit_B, abs(yy), fit_B, abs(test_f));
%   legend({'real', 'imag', 'fit real', 'fit imag', 'abs'})
else
  [shf_model, shf_pars] = cw_shf_model(spin_probe);
  [fit.pattern, resolution] = cw_shf(shf_pars);
%   fit.Bshf = (1:length(fit.pattern))' * resolution;
  fit.Bshf = (1:length(fit.pattern))' * resolution * 2;
  fit.Bshf = fit.Bshf - mean(fit.Bshf);
  fit.B   = fit_B;
  
  fmin = @(x) norm(yy - simple_fit(x,fit));
  fmin_abs = @(x) norm(abs(yy) - simple_fit_abs(x,fit));
  x4 = fminsearch(fmin_abs, [max(abs(y)), mean(x), 10e-3]); x4(4) = 0;
  x2 = fminsearch(fmin, x4);
  
  fit.B = x; out_fit = simple_fit(x2, fit);
end
ph = kphase*x2(4)*180/pi;
x_0 = x2(2);
lw = x2(3);

% figure(505); clf;
% subplot(2,1,1); hold on;
% iplot(x(idx),y(idx),1); iplot(x(idx),out_fit(idx),3)
% axis tight; grid on
% text(0.05,0.8,sprintf('range = %4.3f G',max(x) - min(x)),'units','normalized')
% text(0.8,0.8,sprintf('ph = %4.2f',ph),'units','normalized')
% text(0.8,0.7,sprintf('x_0 = %5.3f G',x2(2)),'units','normalized')
% text(0.8,0.6,sprintf('lw = %4.2f mG',lw*1E3),'units','normalized')
% subplot(2,1,2)
% iplot(x,y-out_fit)
% axis tight; grid on

% --------------------------------------------------------------------
function y = simple_fit(x, pars)

lw = epr_Lorentzian(pars.Bshf, 0,abs(x(3)));
y = conv(lw,pars.pattern,'same');
y = abs(x(1))*interp1(pars.Bshf+x(2), y, pars.B,'spline',0)*exp(-1i*x(4));

% --------------------------------------------------------------------
function y = simple_fit_abs(x, pars)

lw = epr_Lorentzian(pars.Bshf, 0,abs(x(3)));
y = conv(lw,pars.pattern,'same');
y = abs(x(1)*interp1(pars.Bshf+x(2), y, pars.B,'spline',0));
