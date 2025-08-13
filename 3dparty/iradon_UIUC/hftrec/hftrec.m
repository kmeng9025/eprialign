function zout = hftrec(zin,npt,iorig,itmx,ictrl);
%
%function zout = hftrec(zin,npt,iorig,itmx,ictrl)
%
%	Perform half Fourier reconstruction along the leading dimension.
%       zin:     input asymmetric echos
%       npt:     number of points for the symmetrized echoes
%       iorig:   echo location
%       itmx:    number of iterations (default: itmx=4)
%       ictrl:   symmetrized data (0); reconstructed image (1; default)
%
%
if (nargin < 4) 
   itmx = 4;
   ictrl =1;
end

if (nargin < 5) 
   ictrl =1;
end

dim    = size(zin);
nview  = prod(dim)/dim(1);

zin  = reshape(zin,[dim(1),nview]);
zout = zeros([npt,nview]); 
ir   = npt/2+1-iorig;
ncenter = 2*(iorig-1);
filter = ones(ncenter,1);
for nn=1:ncenter
    filter(nn) = 0.6+0.4*cos(2*pi*(nn-iorig)/ncenter); 
end
for nn = 1:nview
    phas = zeros([npt,1]);
    phas(ir:ir+ncenter-1) = zin(1:ncenter,nn).*filter(:);
    phas = Pishft1(phas,1);
    phas = ifft(phas);
    phas = Pishft1(phas,1);
    phas = angle(phas);
    zout(:,nn) = pocs(zin(:,nn),phas,iorig,itmx); 
    zout(ir+dim(1):npt,nn) = 0;
    zout(1:npt-ir-dim(1)+1,nn) = 0;
end
if (ictrl == 1)
   zout = Pishft1(zout,1);
   zout = ifft(zout);
   zout = Pishft1(zout,1);
end

zin    = reshape(zin,dim);
dim(1) = npt;
zout   = reshape(zout,dim);

return;
