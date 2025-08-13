function [res,err]=deco(sp,krn,coef)

% zero padding
s=size(sp);z=zeros(s); inx=[z abs(sp)+1 z]>0.5;
sp=[z sp z];
krn=[z krn z];


KERN=fft(fftshift(krn));
S=length(KERN); 


SP=fft(sp); 
FILTER=conj(KERN)./( abs(KERN).^2+coef*hamming(S)');
%FILTER2=1./( KERN+coef);

%RES=SP./(KERN+coef);
RES=SP.*FILTER;

SPtest=RES.*KERN;
spTest=real(ifft(SPtest));
d=sp-spTest;
err=d*d';
ERR=sum(abs(SPtest-SP))/length(inx);
res=ifft(RES);
res=res(inx);
res=real(res);

% res=res/sum(res)*sum(sp);