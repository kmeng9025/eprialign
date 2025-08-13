function [tx,Mx,My,Mz]=RS_Bloch(TT,xAm,xFm,dH)
%T2 s
%Am - mudulation amplitude, G
%Fm - modulation frequency, Hz
%dH - magnetic field offset, G
I=sqrt(-1);
err=1e-11; gamma=1.76e7; % rad s-1 G-1
options = odeset('RelTol',1e-11,'AbsTol',[err err err]);
global T1 T2 dw Am Fm Bx
%%
%load Pars P
%%
UC=gamma/2/pi;     % conversion factor G -> Hz
T1=TT(1); 
T2=TT(2);

Am=xAm;     % Mod amplitude G
Fm=xFm;     % Mod frequency Hz
Bx=1e-4;  % G
P=1/Fm; %  Period
T=4*P;

dt=round(1/Fm*1e8)*1e-11;  
dt=T/(4*8192);

%%

t=0:dt:T;
Nt=length(t);
inT=t>T-P;
Mod=cos(2*pi*Fm*t);
tx=t(inT);
tx=tx-tx(1);
plot(t,Mod,tx,Mod(inT));
dw=dH;
       
[Tau,Y]=ode45(@RS,t,[0 0 1],options);
Mx=-Y(inT,1)';
My=-Y(inT,2)';
Mz=Y(inT,3)';
% plt3(Mx,My,(Mz-1)*100);
