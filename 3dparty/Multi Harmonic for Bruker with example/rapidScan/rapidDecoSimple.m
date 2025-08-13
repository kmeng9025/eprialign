function res=rapidDecoSimple(Bm,Fm,RapidSp)
% Bm=1.839; % SWEEP WIDTH
% Fm=8020; % SWEEP FREQUENCY
M=length(RapidSp)/2; %number of points in projection
% GENEARATE ANALYTICAL FUNCTION
k       = 1:M;                                 % indeces
b       = 2.8E6 * 2 * Bm * 2 * pi * Fm;         % Scan Rate
d		= 1/(b);
domega	= (2 * 2*pi*Fm);
w		= k * domega;
A = exp(-1i*(d*w.^2*1/2 )); % analytical function
opposite=conj(A);                              % CHANGE FOR EVERY PROGRAM DECONVOLUTION
A((M+1):2*M)=opposite(M:-1:1);

h=fft(RapidSp);         
SU=h./A;                                                % DIVIDING BY ANALYTICAL EXPRESSION
res=ifft(SU,2*M);
