function x=xForInterp(sweep,N)

x=(0:(N-1) )/(N-1);
sw2=sweep/2;
x=-sw2+x*sweep;