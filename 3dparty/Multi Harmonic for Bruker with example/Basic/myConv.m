function res=myConv(data,kern,zp)
% zp - zeropadding
L=length(data);
ss=size(data);
x=zeros(ss);
x=[x; x];
if zp==0; x=[]; end;
if ss(1)==1
    data=[x data x];
    kern=[x kern x];
end

if ss(2)==1
    data=[x; data; x];
    kern=[x; kern; x];
end
% nnn=Round[FirstMom[data]];
tmp=fft(data);
tmp2=fft(fftshift(kern));
tmp3=(ifft(tmp.*tmp2));

if zp==0;
    res=tmp3;
else
    res=tmp3(2*L+1:3*L);
end



%
% x=zeros(ss);
%
% I=sqrt(-1);
% xxx= 2*ifft(I*sp);
%
% L=length(xxx);
%
% xxx(1:round(L/2))=0;
% res=fft(xxx);
%
