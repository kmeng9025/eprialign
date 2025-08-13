% P- set of projections
function [bx,mi,xshift]=ShiftNext1D(an, bm, n, m)
clc;
a=an; % n-th  projection
b=bm; % m-th  projection ! to be corrected (shifted)
%imagesc([a b]);

mi=100000000;
xshift=0;
for shift=-100:10:100     %%%  ???                     !!!!
ma=max(abs(a))/10; i=abs(a)<ma; a(i)=0;
mb=max(abs(b))/10; i=abs(b)<mb; b(i)=0;
bx=rotatev(b',shift);
tmp=(a-bx');  z=tmp'*tmp/length(tmp)/length(tmp);
if z<mi
    mi=z;
    xshift=shift;
end
end
bx=rotatev(b',xshift);
% imagesc([a bx']);

%shift=WhatIsShift(a,b);
% b1=rotatev(b',-shift);
% imagesc([a b1']);


