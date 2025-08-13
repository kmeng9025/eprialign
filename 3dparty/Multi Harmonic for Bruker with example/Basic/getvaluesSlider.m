%totalL=50;

slid=get(s1,'Value');
ss=size(newP);
Newslid=round(ss(2)*slid);
if Newslid==0 Newslid=1; end;
NewsliS=num2str(Newslid);

set(t1,'String',NewsliS);
%% SHOW A/B
tnp=newP(:,Newslid);
 tnp=zeroLine(tnp',0.05);  tnp=zeroLine(tnp,0.05);
plot(tnp);

[mx,i]=max(tnp); [mn,j]=min(tnp);
ab=-mx/mn;
intensity=mx-mn;
dHx=round((i-j)/ss(1)*dH);

abS=num2str(ab); dHs=num2str(dHx); intenS=num2str(intensity); str= [dHs ,'G, A/B= ' ,abS,' A+B=', intenS];
set(t3x,'String',str);

% shfta=round((get(s1,'Value')-0.5)*sweep);
% shftb=round((get(s2,'Value')-0.5)*sweep);
% 
% a=projs(:,num)';
% b=projs(:,totalL+1-num)';
% 
% shA=rotatev(a,shfta+shftb);
% shB=fliplr(rotatev(b,shfta-shftb));
% new=integ(real(fromImToComplex((shA+shB))));
% 
% axes(axe1); plot(shA);
% axes(axe2); plot(shB);
% axes(axe3); plot(new);

