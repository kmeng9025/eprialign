function y = epr_Lorentzian(x, x0, fwhm)
xc = 2*(x-x0)/fwhm;
y = 2/pi/fwhm.*(1-1i*xc)./(1+xc.*xc);