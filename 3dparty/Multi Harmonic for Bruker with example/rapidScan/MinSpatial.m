function [Sp,test1]=MinSpatial(D,re1,LL,D1,W,st1,deltaH,theta,M,Z,dz);
Nimage=D.Nimage;

% Sp - spatial profile
% tst - Smoot test in %

Z1=Z(1);
Z2=Z(2);

LZ=length(Z);
if LZ>2
   n=(LZ-2)/2;
   jnx=ones(Z2-Z1+1,1);
   for i=1:n
       iz=[Z(2+i*2-1) Z(2+2*i)]-Z(1);
        jnx(iz(1):iz(2))=0;
   end
end

dRm=LorRadon(W,deltaH,theta,M,[Z1 Z2]);
RR=dRm'*dRm;
f=dRm'*re1;


list1=10.^(-3:1:3);
C=length(list1);
test=zeros(1,C);
err=test;




TMP=zeros(C,Nimage);
% Sp0=Sp;
for k=1:C
    F=list1(k)*LL+RR;
    F=inv(F);
    Sp=F*f;
    Sp=Sp.*jnx;
    in=Sp<0;  Sp(in)=0;
    test(k)=(sum(abs(D1*Sp))/(length(Sp)*dz)-st1)/st1;
    Sp=[zeros(1,(Z1-1)) Sp' zeros(1,Nimage-Z2)];
    P=LorRadonSp(Sp,W,deltaH,theta,M,[Z1 Z2]);
    R=sum(P')';
    dR=re1-R;
    err(k)=sum(dR.^2);
    TMP(k,:)=Sp;
end


[V iee2]=min(err);
lmd1=list1(iee2);
Err=err(iee2);
Sp=TMP(iee2,:);
test1=test(iee2);
