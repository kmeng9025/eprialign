function res=antiQeffect(t,B1,Q,v0)

[w,FB1]=fftM(t,B1);
X=2*pi*w*Q/(pi*v0);
F=(1-1i*X)./(1+X.^2);
%HPF=hann(length(FB1))';
Fmax=max(abs(F));
inx=abs(F)<0.5*Fmax;
F(inx)=Fmax;
%FB1=FB1.*HPF';
xxx=FB1./F;
xxx=ifftshift(xxx);
res=ifft(xxx);
%B10=real(B10);
plot(t,real(res),t,imag(res),'-');
%axis([0 1.5e-6 -.00001 .00001])