% P- set of projections
% подгон*ет сдвига* и мен** маштаб

clear;
load data;
load par ;
% function Px=ShiftNext(P, n, m)
sw=300;
Sweep=sw*grad;

Px=data2D; n=1 ; m=2;
P=Px; 

tmp1=P(:,n);
tmp2=P(:,m); % подгон*ется

oldSweep = Sweep(n);
newSweep = Sweep(m); % здесь меньший
 forN=10;
 MMM=10000000000;
 for i=0:forN
     delta=i * (oldSweep-newSweep)/10;
     tmp3=reSweep(tmp2,newSweep+delta,oldSweep);
     [xxx,mi,shift]=ShiftNext1D(tmp1,tmp3, n, m);
        if mi<MMM
            MMM=mi;
            shftG=newSweep+delta;
            P(:,m)=rotatev(tmp2,shift);
        end
 end;

%
 
 imagesc([P(:,n) P(:,m)]);

%shift=WhatIsShift(a,b);
% b1=rotatev(b',-shift);
% imagesc([a b1']);


