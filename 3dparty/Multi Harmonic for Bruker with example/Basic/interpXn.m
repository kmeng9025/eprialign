function r=interpXn(y,N)
L=length(y);
x=0:(L-1);
x=x/max(x);

z=0:(L*N-1);
z=z/max(z);
%plt2(x,z)
r=interp1(x,y,z);