function Px=ShiftAll(P,num,dir);
ss=size(P);
Px=P;
s2=ss(2);

if dir>0
  for n=num:(s2-1)
   Px=ShiftNext(Px, n, n+1);
  end;
end

if dir<0
    
 for n=num:(-1):2
   Px=ShiftNext(Px, n, n-1);
  end;
end

%shift=WhatIsShift(a,b);
% b1=rotatev(b',-shift);
% imagesc([a b1']);