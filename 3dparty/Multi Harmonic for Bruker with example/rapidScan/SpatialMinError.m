dRm=LorRadon(W,deltaH,theta(inx1),M,[Z1 Z2]);
RR=dRm'*dRm;
f=dRm'*re1;
%% - SPATIAL ---------------------------------------
test=zeros(1,C);
err=test;


lambda=lmd1*list;

TMP=zeros(C,Nimage);
Sp0=Sp;
for k=1:C
    F=lambda(k)*LL+RR;
    F=inv(F);
    Sp=F*f;
    in=Sp<0;  Sp(in)=0;
    test(k)=(sum(abs(D1*Sp))/(length(Sp)*dz)-st1)/st1;
    Sp=[zeros(1,(Z1-1)) Sp' zeros(1,Nimage-Z2)];
    P=LorRadonSp(Sp,W,deltaH,theta(inx0),M,[Z1 Z2]);
    R=sum(P')';
    dR=re-R;
    err(k)=sum(dR.^2);
    TMP(k,:)=Sp;
end
iee=err<Err;
if sum(iee)>0;
    [V iee2]=min(err);
    lmd1=lambda(iee2);
    Err=err(iee2);
    Sp=TMP(iee2,:);
    test1=test(iee2);
else
    Sp=Sp0;
end
