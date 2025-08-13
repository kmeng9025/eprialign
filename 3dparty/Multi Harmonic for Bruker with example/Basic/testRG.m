x=-6:.01:6;
R=20;
g=1;
psi=g*0.02/R;

[x1,y]=dysonxRsur(x,R,psi);

plot(x1,y);
