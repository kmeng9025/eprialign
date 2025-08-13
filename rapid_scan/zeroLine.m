function [res,tmp]=zeroLine(spectrum,extent)
L=length(spectrum);
edge=round(extent*L);
if edge==0; edge=1; end;

Sleft =  sum(  spectrum(1:edge)     )/edge;
Sright = sum(  spectrum(L-edge+1:L) )/edge;
s=size(spectrum);

%for i=1:L
tmp=Sleft+(Sright-Sleft)/L*(1:L);
if size(tmp)~=s; 
    tmp=conj(tmp');
end;
%end

res=spectrum-tmp;
