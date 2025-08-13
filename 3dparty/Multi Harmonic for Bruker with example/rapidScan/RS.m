function dM = RS(t,M)
% global X Y D h f A WF;
gamma=1.76e7; % rad s-1 G-1
global T1 T2 dw Am Fm Bx
% --------------------- %
wy=0;
wx=gamma*Bx;
A=gamma*(dw+0.5*Am*cos(2*pi*Fm*t));
% ------------------%
dM = zeros(3,1);   % (x,y,z) a column vector
M0=1;
%!!!   dw=w-w0;
%          M1         M2         M3          M0
dM(1) = -M(1)/T2    +A*M(2)   +wy*M(3)    +0;
dM(2) = -A*M(1)    -M(2)/T2   -wx*M(3)    +0;
dM(3) = -wy*M(1)    +wx*M(2)   -M(3)/T1    +M0/T1;

% A=[-1/T2  -dw     By; ...
%     dw   -1/T2   Bx;
%    -Bx    -By   -1/T2];
% b=[0; 0 ;M0/T1];
% dM=A*M+b;
%     
