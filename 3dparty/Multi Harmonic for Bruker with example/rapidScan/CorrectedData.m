function [res err]=CorrectedData(fn,fi,trigger,sweep,freq,cycles,comm);
% fn-  Bruker file name
% fi - phase correction
% trigger - trigger correction
% sweep -   sweep Widths
% freq - frequency
% cycles - number of full cycles in the data

% load tmp; 
% trigger=trigger-0.2;

%%
n=8000;
data=importXEPR1(fn);
sp=data.intensity;
N=length(sp);
sp=interpNM(sp,4*N);

tb=round(data.xWIDTH/data.xPTS)*10^(-9); % time base
comment=data.TITLE;
%%
T2=1/freq/2; ;
nPoints=round(8*T2/tb);
p1=round(trigger/360*nPoints);
p2=p1+nPoints;
t=1:(nPoints-1)/(n-1):nPoints;





% break
sp1=sp(p1:p1+nPoints-1);
sp1x=interp1(1:nPoints,sp1,t);
sp1y=sp1x*exp(sqrt(-1)*fi );
sp1=rapidDecoC(sweep,freq/4,sp1y);
L=length(sp1);

S1=sp1(1:L/2);  S1f=fliplr(sp1(L/2+1:L));
Sr1=(S1+S1f)/2;

if cycles==2;
    sp2=sp(p2:p2+nPoints-1);
    sp2x=interp1(1:nPoints,sp2,t);
    sp2y=sp2x*exp(sqrt(-1)*fi );
    sp1=rapidDecoCZ(sw,f/4,sp2y);
    S2=sp1(1:L/2);  S2f=fliplr(sp1(L/2+1:L));
    Sr2=(S2+S2f)/2;
end;
  
%% --   PLOT   ------------------
subplot(4,1,1); %1
plot(real(sp)); hold on;
inx=p1:p1+nPoints-1;
plot(inx,real(sp(inx)),'-r');
if cycles==2; plot(p2:p2+nPoints-1,real(sp2),'-g'); end;
title(['Exper' comm ' ', fn ',  Triger position=' num2str(trigger)]);
hold off; 
axis tight;


subplot(4,1,[2 3 4]); %1
plot(real(S1)); hold on;
plot(real(S1f),'-r'); 
delta=real(S1f-S1);
plot(delta,'-g'); 
hold off;

err=sum(delta.^2)/length(delta);
if cycles==1 
   res=Sr1;
else
   res=[Sr1; Sr2];
end


