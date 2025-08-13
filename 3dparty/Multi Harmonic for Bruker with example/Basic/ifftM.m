function [w,A]=ifftM(t,a)
a=ifftshift(a);
%a=ifftshift(ifft(a));

dt=t(2)-t(1);
L=length(t);
Wmax=1/dt;
T=dt*(L-1);
dw=1/T;
Left=-Wmax/2;
Right=Wmax/2-dw;
dw1=(Wmax-dw)/(L-1);
w=Left:dw1:Right;
A=ifft(a);
%A=ifftshift(A);