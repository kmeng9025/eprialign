% EPRI_LSHAPE
% y = epri_lshape(x, x0, fwhm, alpha)
function y = epri_lshape(x, x0, fwhm, alpha)
if nargin < 4, alpha = -1; end
% Lorentzian
if fwhm(1) == 0
    xlc = 2*(x-x0)/fwhm(2);
    y = 2/pi/fwhm(2).*(1-1i*xlc)./(1+xlc.*xlc);
elseif fwhm(2) == 0
    % Gaussian
    % fwhm = 2*sqrt(2*log(2))*sigma; = 2.354820045030949 * sigma
    % 2.354820045030949/sqrt(2*pi) = 0.939437278699651
    xgc = (x-x0)/fwhm(1)*2.354820045030949;
    y = exp(-xgc.*xgc/2).*(1-1i*xgc)/fwhm(1)*0.939437278699651;
    % val=pars(3)*1/sqrt(2*pi*pars(2))*exp(-(xdata-pars(1)).^2/(2*pars(2)));
elseif alpha ~= -1
    % Pseudo-Voight
    y = (1-alpha)*epri_lshape(x, x0, [fwhm(1), 0]) + alpha*epri_lshape(x, x0, [0, fwhm(2)]);
else
    % convolution of Gaussian and Lorentzian
    xlc = x - (max(x)+min(x))/2;
    yl = epri_lshape(xlc, 0, [0 fwhm(2)]);
    xgc = (x-x0)/fwhm(1)*2.354820045030949;
    yg = exp(-xgc.*xgc/2)/fwhm(1)*0.939437278699651;
    y = conv(yg,yl,'same');
end

% figure(10); clf; x = -300:300; y = epri_lshape(x, 25, [20,20], 0.5); plot(x, real(y), x, imag(y)); disp(sum(y))
