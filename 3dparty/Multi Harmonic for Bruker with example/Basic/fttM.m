function [w,A]=fftM(t,a);
dt=t(2)-t(1);
L=length(t);
Wmax=1/dt;
T=dt*(L-1);
dw=1/T;
w=-Wmax/2:dw:Wmax/2;
A=fft(a);