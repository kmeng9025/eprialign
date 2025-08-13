function [Z W]=LorDifRadon(Spatial,widths,deltaH,theta,size,y12)
% Z - derivative of Radon
% W- Radon
% y12(1) start index for spatial profile
% y12(2) finish index
% spectral profile vector
% theta - angles in degrees
% size - points in one projection
% deltaH- spectral window of image

M=length(widths);
G=length(theta);
widths=widths(y12(1):y12(2));
Spatial=Spatial(y12(1):y12(2));
Mc=length(widths);
Z=zeros(size*G,Mc);
W=Z;
T=repmat(widths,size,1); % vector with N elements
A=repmat(Spatial,size,1); % vector with N elements

for g=1:G;
    t=theta(g)/180*pi;
    sw=sqrt(2)*deltaH/cos(t);
    y=xForInterp(1,M)*tan(t)*deltaH;
    y=y(y12(1):y12(2));
    x=xForInterp(sw,size);
    dx=abs( x(2)-x(1));
    N=length(x);
    X=repmat(x',1,Mc);
    Y=repmat(y,N,1);
    X=X+Y;
    D=1./(T.^2+X.^2);
    B=T./(T.^2+X.^2);
    z=A/pi.*D.*(1-2*T.*B);
    Z(((g-1)*size+1):g*size,:)=z*dx;
    W(((g-1)*size+1):g*size,:)=1/pi*A.*B*dx;
end


