
function zout = pocs(zin,phas,iorig,itmx)
%
%       function zout = pocs(zin,phas,iorig,itmx)
%
%	POCS symmetrization of asymmetric echos
%       zout:  output symmetrized echo 
%       zin:   input asymmetric echo with origin at iorig
%       itmx:  number of pocs iteration
%
n1 = size(zin,1);
n2 = size(phas,1);

ir     = n2/2+1-iorig;
zout   = zeros([n2 1]);
cphase = exp(i*phas(:));
for iter = 1:itmx
    zout(ir:ir+n1-1) = zin;
    zout = Pishft1(zout,1);
    zout = ifft(zout);
    zout = abs(zout).*cphase;
    zout = Pishft1(zout,1);
    zout = fft(zout);
    zout = Pishft1(zout,1);
end
zout(ir:ir+n1-1) = zin;

return;
