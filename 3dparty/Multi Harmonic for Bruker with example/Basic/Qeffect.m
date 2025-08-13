function res=Qeffect(t,B1,Q,v0)

[w,FB1]=fftM(t,B1);
FB1=fftshift(FB1);
X=2*pi*w*Q/(pi*v0);
F=(1-1i*X)./(1+X.^2);
HPF=hann(length(FB1));
FB1=FB1.*HPF';
pltc(FB1);
res=1*(ifft(ifftshift(FB1.*F)));
%B10=real(B10);
plot(t,real(res),t,imag(res),'-');
%axis([0 1.5e-6 -.00001 .00001])