function res=fromImToComplex(spectrum)

l=length(spectrum);
ss=size(spectrum);

if ss(2)<ss(1)
    spectrum=spectrum';
end

    x=zeros(1,l);
    sp=[x x spectrum x x];
    xxx= 2*ifft(1i*sp);
    L=length(xxx);
    xxx(1:round(L/2))=0;
    res=fft(xxx);
    res=res(2*l+1:3*l);


