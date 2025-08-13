function res=rapidDecoSawToothZL(Bm,Fm,RapidSp,ZL);
% Bm=1.839; % SWEEP WIDTH
% Fm=8020; % SWEEP FREQUENCY
s=size(RapidSp);

if s(1)>s(2); RapidSp=RapidSp'; end;

n1=length(RapidSp); %number of points in projection
if ZL==1
    
    rx=real(RapidSp);
    ry=imag(RapidSp);
    gx=zeroLineD(rx,0.05); 
    gy=zeroLineD(ry,0.05);
    g=gx+sqrt(-1)*gy;
    tmp=RapidSp-g;
else
    g=RapidSp;
end
Q=2.^(10:20);
inx=(n1<Q);
QQ=Q(inx);
N=min(QQ);
i=(0:(N-1))/(N-1);
M=length(g);
j=(0:(M-1))/(M-1);
gx=interp1(j,g,i);

g=gx;
% g(n1:N) = zeros;

% g=conj(g);
h=fft(g,N);                                       %FFT 
n2=N;

% GENEARATE ANALYTICAL FUNCTION
I=sqrt(-1);
k       = 1:n2;                                 % indeces
b       = 2.8E6 * 2 * Bm * 2 * pi * Fm;         % Scan Rate
d		= 1/(b);
domega	= (2 * 2*pi*Fm) * n1 / n2;
% df      = 2.8E6 * Bm / n1;                      % frequency step
%fwid    = df * n2 ;                             % frequency bin width
%dt      = 1 / fwid;                             % time step
%t       = k * dt;                               % time points
w		= k * domega;
A = exp(-I*(d*w.^2*1/2 )); % analytical function



opposite=conj(A);                              % CHANGE FOR EVERY PROGRAM DECONVOLUTION
A((N/2+1):N)=opposite(N/2:-1:1);
%step=2*Fm*n1/n2;   % MULTIPLY BY SOME NUMBERS TO GET CUT OFF FREQUENCY POINT
SU=h./A;                                                % DIVIDING BY ANALYTICAL EXPRESSION
SA=ifft(SU,N);
%INVERSE FOURIER TRANSFORM OF DECONVULTED SIGNAL

% JH=SA(1:n1);
% if ZL==1
%     res=real(JH+tmp);
% else
%     res=real(JH);
% end
res=SA;
%res=real(JH);

% plot(JH)