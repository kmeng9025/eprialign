function res=decoWiner(sp,krn,N);

% zero padding
s=size(sp);z=zeros(s); inx=[z abs(sp)+1 z]>0.5;
sp=[z sp z];
krn=[z krn z];


H=fft(krn);
B=fft(sp); 

xxx=(B-N); F2=xxx.*conj(xxx);
C=B.*F2;
D=H.*(F2+N^2);
F=C./(1+D);
res=imag(ifft(F));
res=res(inx);
plot(abs(F))