% out = epr_lowpass(dt, in, pass)
% FT filter

function out = epr_lowpass(dt, in, pass)

fax = linspace(-1/2/dt, 1/2/dt, length(in));
in_ft = fftshift(fft(in));

in_ft(abs(fax) > pass) = 0;
out = ifft(fftshift(in_ft));

