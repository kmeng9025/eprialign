function [res,tmp]=zeroLineD(spectrum,extent)
L=length(spectrum);
edge=round(extent*L);

Sleft =  sum(  spectrum(1:edge)     )/edge;
Sright = sum(  spectrum(L-edge+1:L) )/edge;
s=size(spectrum);

%for i=1:L
tmp=(Sright+Sleft)/2;

res=spectrum-tmp;
