 function newm=centering(m,a);
  % m- vector
  % a -accuracy a it better use a not much than 3,4,5
  % [2 2 3 3]  ->  [2 3 3 2]
  % 
 [momleft,momright]=moments1(m); %moments of two half of m left and right 
 s=sum(m);
 shft=round((momright-momleft)/s);
 dmom=100000;
 dk=0;
 for k=-a:a
 nw=rotatev(m,shft+k);
 [momleft,momright]=moments1(nw);
 d=abs((momright-momleft)/s);
 if d<dmom
    dmom=d; 
    dk=k;
    newm=nw;
 end
 end
