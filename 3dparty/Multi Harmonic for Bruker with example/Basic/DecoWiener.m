function  res=DecoWiener(krn,convoluted,noise);
% out - convoluted spectrum
% kern - point spread function
% tol - tolirance // for pseudo -inv- matrix
% epsilon  // regularization parameter

% len=length(krn);
sp=convoluted;

s=size(sp);z=zeros(s); inx=[z abs(sp)+1 z]>0.5;
sp=[z sp z];
krn=[z krn z];
noise=[z noise z];

Ns=fft(noise);
KERN=fft(fftshift(krn));
SP=fft(sp); 
% FILTER=conj(KERN)./( abs(KERN).^2+coef);
% FILTER2=1./( KERN+coef);
SPabs=abs(SP);
Ns=abs(Ns);
Ns=0.01*max(Ns);

tmp=abs(SP)-Ns; i=tmp<0; tmp(i)=0;
Fltr=tmp./abs(SP)
%RES=SP./(KERN+coef);
RES=(SP.*Fltr)./KERN;

res=ifft(RES);
res=res(inx);
