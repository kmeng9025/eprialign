function P = epr_deconvolve(x,y,xref,ref,pars)

ntrace = size(y,1); 
nslice = size(y,2); 
xfourier = linspace(0,2*ntrace,ntrace)'; 

P = zeros(size(y));

switch upper(safeget(pars, 'function', 'GAUSSIAN'))
  case 'GAUSSIAN'
    fwhm = safeget(pars, 'fwhm', 1);
    filter = sqrt(2/pi)*(1/(fwhm/sqrt(2*log(2))))*exp(-2*((xfourier-ntrace)/(fwhm/sqrt(2*log(2)))).^2);
  case 'FERMI'
    Fermi_radius = safeget(pars, 'radius', 1);
    Fermi_roll   = safeget(pars, 'roll', 1);
    filter(xfourier >= ntrace) = 1./(exp((xfourier(xfourier >= ntrace)-ntrace-Fermi_radius)/Fermi_roll)+1);
    filter(xfourier < ntrace) = 1./(exp((-xfourier(xfourier < ntrace)+ntrace-Fermi_radius)/Fermi_roll)+1);
  otherwise
    filter = ones(size(xfourier)) / (2*ntrace);
end

% figure(6); plot(filter); axis tight

h = fftshift(filter(:))./fft(ref);

for k=1:nslice 
    raw_fft = ifftshift(ifft(fft(y(:,k)).*h));
    P(:,k)=real(raw_fft);
end
