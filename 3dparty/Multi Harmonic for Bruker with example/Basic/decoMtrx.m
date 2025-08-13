% function res=decoMtrx(sp,krn,coef);
clc
coef=.01;
sp=M2;
krn= M2y0;
s=size(M2); st=s(1);sf=s(2); 

SP=fft(sp');

% zero padding
% s=size(sp);z=zeros(s); inx=[z abs(sp)+1 z]>0.5;
% % sp=[z sp z];
% % krn=[z krn z];
% 
% 
KERN=fft(fftshift(krn'));
HAM=repmat(hamming(sf),1,st);
% fig2(abs(KERN),HAM);
%fig2(HAM,sp)
% whos SP KERN
% S=length(KERN); 
% SP=fft(sp); 
FILTER=conj(KERN)./( abs(KERN).^2+coef*HAM);
% fig2(abs(FILTER),abs(KERN))
% %FILTER2=1./( KERN+coef);
% 
% %RES=SP./(KERN+coef);
RES=SP.*FILTER;
% fig2(abs(RES),abs(SP))
res=ifft(RES)/st;
% res=res(inx);
res=real(res);
fig2(res,res)% res=res/sum(res)*sum(sp);
res=res(:,:);
plot(sum(res'))

