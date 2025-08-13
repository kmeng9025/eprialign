function [W,test2]=MinWidthsFlat(D,re2,LL,D1,Sp,W,st2,deltaH,theta,M,Z,dz,TH,jnx);
% global Nimage UpperLim LowerLim
Nimage=D.Nimage; Lmodel=D.Lmodel; 
LowerLim=D.limits(1);  UpperLim=D.limits(2); deltaH=D.deltaH;
% Sp - spatial profile
% tst - Smoot test in %
%   === Spectral profile ======------------------------------------
% TH - threshold for dw;

Z1=Z(1); Z2=Z(2);
Z3=Z(3); Z4=Z(4);
Z5=Z(5); Z6=Z(6);
in1=1:(Z3-Z1);
in2=(Z4-Z1):(Z5-Z1);
in3=(Z6-Z1):(Z2-Z1+1);
[Q P]=LorDifRadon(Sp,W,deltaH,theta,M,[Z1 Z2]);

Q1=sum(Q(:,in1)');
Q2=sum(Q(:,in2)');
Q3=sum(Q(:,in3)');
Qx=[Q1; Q2; Q3]';

R=sum(P')';
dR=re2-R;
QQ=Qx'*Qx;
Wc=W(Z1:Z2);

list1=10.^(-3:1:1);
C1=length(list1);
test=zeros(1,C1);
TMP=zeros(C1,Nimage);
W0=W;

dwx=inv(QQ) * (Qx'*dR);
dw=0*Wc;
dw(in1)=dwx(1);
dw(in2)=dwx(2);
dw(in3)=dwx(3);
  
%W=W+dw;

% ZZ=Z2-Z1+1;
% % ZZ=ZZ-sum(iSp(Z1:Z2));
% Den=ZZ*dz;
% for k=1:C1
%     dw=inv(QQ+list1(k)*LL) * (Q'*dR-list1(k)*LL*Wc');
     ratio=dw./Wc;
     ir=abs(ratio)>TH;
     dw(ir)=TH*sign(ratio(ir)).*Wc(ir);
     dw=[zeros((Z1-1),1); dw'; zeros(Nimage-Z2,1)];
     W=W0+dw'; %!!
%    
     inx=W>UpperLim ;  W(inx)=UpperLim;
     inx=W<LowerLim ;  W(inx)=LowerLim;
%%     test(k)=testW(W,D1,Sp,st2,jnx,dz,Z);
%     
%     P=LorRadonSp(Sp,W,deltaH,theta,M,[Z1 Z2]);
%     R=sum(P')';
%     dR=re2-R;
%     err(k)=sum(dR.^2);
%     TMP(k,:)=W;
% end
% 
% 
% [V iee2]=min(err);
% lmd1=list1(iee2);
% Err=err(iee2);
% W=TMP(iee2,:);
test2=0;
