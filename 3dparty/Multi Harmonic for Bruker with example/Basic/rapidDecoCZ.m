function res=rapidDecoCZ(Bm,Fm,RapidSp);
% Bm=1.839; % SWEEP WIDTH
% Fm=8020; % SWEEP FREQUENCY
n1=length(RapidSp); %number of points in projection

rx=real(RapidSp);
ry=imag(RapidSp);

gx=zeroLine(rx,0.05);
gy=zeroLineD(ry,0.05);
g=gx+sqrt(-1)*gy;
gx=zeroLine(RapidSp,0.05);
delta=RapidSp-g;

M=2*2^( ceil( log2(n1)) );

g(n1:M) = zeros;

% g=conj(g);
h=fft(g,2*M);                                       %FFT 
n2=2*M;

% GENEARATE ANALYTICAL FUNCTION

k       = 1:n2;                                 % indeces
b       = 2.8E6 * 2 * Bm * 2 * pi * Fm;         % Scan Rate
d		= 1/(b);
domega	= (2 * 2*pi*Fm) * n1 / n2;
df      = 2.8E6 * Bm / n1;                      % frequency step
fwid    = df * n2 ;                             % frequency bin width
dt      = 1 / fwid;                             % time step
t       = k * dt;                               % time points
w		= k * domega;
A = exp(-1i*(d*w.^2*1/2 )); % analytical function



opposite=conj(A);                              % CHANGE FOR EVERY PROGRAM DECONVOLUTION
A((M+1):2*M)=opposite(M:-1:1);
step=2*Fm*n1/n2       ;   % MULTIPLY BY SOME NUMBERS TO GET CUT OFF FREQUENCY POINT
SU=h./A;     
%SU((M+1):2*M)=0;
%SU(1:M)=0;
SA=ifft(SU);
%INVERSE FOURIER TRANSFORM OF DECONVULTED SIGNAL
JH=SA;
JH=JH(1:n1)+0*delta;
res=JH;
% plot(JH)