% Dysonian Line with 
% GRADIENT, DIFFUSION , SURFACE RELAXATION
% SEE SHOWSPECTRUM 
function [dX,res]=spectrum(g,b1,M,R,s)


% M=50000;
 h=b1/M;   h2=h*h;
% R=.4;  
R2=R*R;
y1=0:h:b1;


g1=abs(g)
% g=-7;
I=sqrt(-1); 
H=exp(-y1*(1+I));  % plot(H,'Color','red'); % 1
H=H+fliplr(H);

left=-(20+b1*g1);
right=g1*b1+8;
step=(right-left)/711;
dX=left:step:right;

Lx=length(dX);
for i=1:Lx

dx=dX(i);
X=My(g,dx,H,b1,M,R,s);
P(i)= sum(H.*X);
i=i+1;
end 
imP=imag(P);
% dP/dx
L=length(P);
for i=1:(L-1)
    dP(i)=imP(i+1)-imP(i);
end
    dP(L)=dP(L-1);
%
res=-dP/step;
% 
% AB=max(dP)/min(dP);
%  plot(-dX,dP);

