function res=fromImToComplex(spectrum)
l=length(spectrum);
mn=mean(spectrum);
spectrum=spectrum -mn;
ss=size(spectrum);
if ss(2)<ss(1)
    x=zeros(ss);
    sp=[x; spectrum; x];
    I=sqrt(-1);
    xxx= 2*ifft(I*sp);
    L=length(xxx);
    xxx(1:round(L/2))=0;
    res=fft(xxx);
    res=res(l+1:2*l);
else
    spectrum=spectrum';
    ss=size(spectrum);
    x=zeros(ss);
    sp=[x; spectrum; x];
    I=sqrt(-1);
    xxx= 2*ifft(I*sp);
    L=length(xxx);
    xxx(1:round(L/2))=0;
    res=fft(xxx);
    res=res(l+1:2*l);
    res=res';
end
res=res +sqrt(-1)*mn;