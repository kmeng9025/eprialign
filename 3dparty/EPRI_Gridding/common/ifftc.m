function [ output ] = ifftc( input)
%ifftc performs center adjusted inverse fourier transform.
output = size(input,1) * fftshift(ifftn(ifftshift(input)));

end

