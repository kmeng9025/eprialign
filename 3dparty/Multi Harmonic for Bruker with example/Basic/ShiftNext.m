% P- set of projections
function [Px,mi]=ShiftNext(P, n, m)
Px=P;

ss=size(P);
a=P(:,n); % n-th  projection
b=P(:,m); % m-th  projection ! to be corrected (shifted)
b1=b;
%imagesc([a b]);
mi=100000000;
xshift=0;
for shift=-100:1:100     %%%  ???                     !!!!
ma=max(abs(a))/10; i=abs(a)<ma; a(i)=0;
mb=max(abs(b))/10; i=abs(b)<mb; b(i)=0;
bx=rotatev(b',shift);
tmp=(a-bx');  z=tmp'*tmp/length(tmp)/length(tmp);
if z<mi
    mi=z;
    xshift=shift;
end
end
bx=rotatev(b1',xshift);
imagesc([a bx']);
Px(:,m)=bx';


%shift=WhatIsShift(a,b);
% b1=rotatev(b',-shift);
% imagesc([a b1']);


