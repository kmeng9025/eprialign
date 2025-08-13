function [t,mx, my, mz]=blochSin(par)
    
%% Load Rapid Scan
gamma=1.7608e7;
Hm=par.hm/2; %  pp modulation amplitude [G]
Fm=par.Vm;
T1=par.T1;
T2=par.T2;
B1=par.B1;
dH=par.dH;  %  offset from central field position
M0=1;
%% Secondary pars
w1=gamma*B1;
Om=-gamma*Hm;
wm=2*pi*Fm;
dw=gamma*dH;
rate=2*Hm*Fm;
bw=2.8e6*6*rate*T2;
N=10*round(bw/Fm); % points in F-domain

%%
n=-N:N;
xm=(-1+n)*T2*wm; % intermediate vars
xp=(+1+n)*T2*wm;
ym=1./(-1i+xm);
yp=1./(-1i+xp);
z=1./(-1i+n*T2*wm);
r=1./(1+1i*n*T1*wm);
%%
A2=1i*T2*Om^2/4*ym;
A1=1i/2*T2*dw*Om*(ym+z);
tmp=Om^2/4*(ym+yp);
bbb=-T1*w1^2.*r;
A0=-1/T2-1i*n*wm+bbb+1i*T2*(dw^2*z+tmp);
B1=1i/2*T2*dw*Om*(yp+z);
B2=1i*T2*Om^2/4*yp;

f=n*0;
f(N+1)=M0*w1;
%%
Y=pentsolveM(A2,A1,A0,B1,B2,f);
Y=transpose(Y);
y=-ifft(ifftshift(Y));
M=2*N+1;
t=(0:M)/M*(1/Fm);
t=t(1:end-1);
my=(2*N+1)*real(y);
%%
ddd=1i*T2/2.*z;
z=zeros(1,M); z(1:end-1)=Y(2:end);
q=zeros(1,M); q(2:end)=Y(1:end-1);
yyy=Om*(z+q)+2*dw*Y;
X=ddd.*yyy;
x=+ifft(ifftshift(X));
mx=(2*N+1)*real(x);
%%
mmm=n*0; 
mmm(N+1)=M0;
mmm=mmm+w1*T1*Y;
Z=mmm.*r;
z=ifft(ifftshift(Z));
mz=(2*N+1)*real(z);


