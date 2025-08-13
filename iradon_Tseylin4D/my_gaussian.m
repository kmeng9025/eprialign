function y=my_gaussian(h,FWHM,der)
% Normalized to 1;
Hpp=FWHM/sqrt(2*log(2));
A=sqrt(2/pi)/Hpp;

a=2*(1/Hpp).^2;

if der==0
    y=A*exp(-a*h.^2);
end

if der==1
    y=A*exp(-a*h.^2).*(-2*a*h);
end