function [xres,yres]=dysonxRsur(x,R,psi);
 % R=11;  x=-6:0.01:6
le=length(x);

i=sqrt(-1);

a=(1 + x);  b=(1 + x.^2); c=(x.^2 - 1);
d =sqrt(b); f=x+i; R2 = R^2;

Ksi= sign(x).*sqrt(d-1);
Eta= sqrt(d + 1);
g=Ksi+i*Eta;

T1 = (R2*f-1).^(-2);
c1 =( 2+(1+i)*R*psi )^2;
c2=R*(g+i*psi);
c3=-2*R*psi+(1-i)*(R2*f-3);  

L=-T1.*(c1./c2+c3);

y=real(L);
% diff
tmp=y-rotatev(y,1);
tmp(1)=tmp(2);
tmp(le)=tmp(le-1);
y=R2*tmp;
xres=-x;
yres= y;
% plot(xres,yres);
 