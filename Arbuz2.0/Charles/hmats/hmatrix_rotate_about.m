function hrotmat = hmatrix_rotate_about(axis, ang)
% create matrix for rotation by "ang" degrees about the direction
% "axis"
%
% C. Pelizzari
% Nov 07

hrotmat=eye(4,4);

s=sin(ang*pi()/180);
c=cos(ang*pi()/180);
t=1.0-c;

axlen=sqrt(dot(axis, axis));
if (axlen == 0) 
    return;
end

x=axis(1)/axlen;
y=axis(2)/axlen;
z=axis(3)/axlen;

m1=[t*x*x t*x*y t*x*z;...
    t*x*y t*y*y t*y*z;...
    t*x*z t*y*z t*z*z];

m2=[  c -z*s  y*s ;...
    z*s    c -x*s ;...
   -y*s  x*s    c] ;

hrotmat(1:3,1:3)=m1'+m2';