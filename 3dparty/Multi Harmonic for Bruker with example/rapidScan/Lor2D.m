function Z=Lor2D(widths,x)
% z=  2D matrix 
N=length(x);
T=repmat(widths,N,1); % vector with N elements
X=repmat(x',1,N);
z=1/pi*T./(T.^2+X.^2);
dx=abs( x(2)-x(1));
Z=z'*dx;