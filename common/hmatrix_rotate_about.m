function hmat = hmatrix_rotate_about(axis, ang, varargin)
% create matrix for rotation by "ang" degrees about the direction
% "axis"
%
% Aug 08 - add optional point about which to rotate.  if third arg is
% specified and it is a vector, use it as the rotation center.
%
% C. Pelizzari
% Nov 07

center = [0 0 0];
if (nargin > 2)
    invec=varargin{1};
    if (isnumeric(invec) && (prod(size(invec)) ==3)), center = invec; end
end
hrotmat=eye(4,4);
hmat = hrotmat;

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
% this is magic, don't ask...
m1=[t*x*x t*x*y t*x*z;...
    t*x*y t*y*y t*y*z;...
    t*x*z t*y*z t*z*z];
% and this is secret, if I told you I'd have to kill you...
m2=[  c -z*s  y*s ;...
    z*s    c -x*s ;...
   -y*s  x*s    c] ;
% this is my final answer:
hrotmat(1:3,1:3)=m1'+m2';

% rotate about center - first translate "center" to origin, then rotate,
% then translate "center" back to where it was before.

hmat = hmatrix_translate(-center) * hrotmat * hmatrix_translate(center);