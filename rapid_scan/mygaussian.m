function y=mygaussian(h,FWHM)
% Normalized to 1;
Hpp=FWHM/sqrt(2*log(2));
A=sqrt(2/pi)/Hpp;
x=2*(h/Hpp).^2;
y=A*exp(-x);
