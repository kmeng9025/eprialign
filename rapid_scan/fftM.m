function [w,A]=fftM(t,a)
if length(t)>1
    dt=t(2)-t(1);
else
    dt=t;
end
L=length(a);
Wmax=1/dt;

T=dt*L;
dw=1/T;
Left=-Wmax/2;
Right=Left+dw*(L-1);

w=Left:dw:Right;
A=fftshift(fft(a));