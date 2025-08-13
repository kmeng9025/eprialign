% ratio=A/B from R u
%

function ratio=ab_R(R);
x=-4:.005:4;
psi=0;

[xres,yres]=dysonxRsur(x,R,psi);
ab=-max(yres)/min(yres);
ratio=ab;


