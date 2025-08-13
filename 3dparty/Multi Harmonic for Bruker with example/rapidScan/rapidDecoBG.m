function res=rapidDeco(Bm,Fm,RapidSp);
% Bm=1.839; % SWEEP WIDTH
% Fm=8020; % SWEEP FREQUENCY
n1=length(RapidSp); %number of points in projection

g=zeroLine(RapidSp,0.05);
delta=RapidSp-g;

g(n1:4096) = zeros;

% g=conj(g);
h=fft(g,8192);                                       %FFT 
n2=8192;

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
A = exp(-i*(d*w.^2*1/2 )); % analytical function



opposite=conj(A);                              % CHANGE FOR EVERY PROGRAM DECONVOLUTION
A(4097:8192)=opposite(4096:-1:1);
step=2*Fm*n1/n2       ;   % MULTIPLY BY SOME NUMBERS TO GET CUT OFF FREQUENCY POINT
SU=h./A;                                                % DIVIDING BY ANALYTICAL EXPRESSION
SA=ifft(SU,8192);
%INVERSE FOURIER TRANSFORM OF DECONVULTED SIGNAL
JH=SA;
JH=JH(1:n1)+delta;
res=real( JH );
% plot(JH)