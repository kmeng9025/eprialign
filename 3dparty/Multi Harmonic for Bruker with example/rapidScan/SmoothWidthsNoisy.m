function [W,tst]=SmoothWidthsNoisy(D,re2,LL,D1,Sp,Sp0,W,st2,deltaH,theta,M,Z,dz,TH,jnx);
% global Nimage UpperLim LowerLim
Nimage=D.Nimage;  
LowerLim=D.limits(1);  UpperLim=D.limits(2); 

% Sp - spatial profile
% tst - Smoot test in %
%   === Spectral profile ======------------------------------------
% TH - threshold for dw;

 Z1=Z(1); Z2=Z(2);
% LZ=length(Z);
% if LZ>2
%    n=(LZ-2)/2;
%    jnx=ones(Z2-Z1+1,1);
%    for i=1:n
%        iz=[Z(2+i*2-1) Z(2+2*i)]-Z(1);
%         jnx(iz(1):iz(2))=0;
%    end
% end

[Q P]=LorDifRadon(Sp,W,deltaH,theta,M,[Z1 Z2]);
R=sum(P')';
dR=re2-R;
QQ=Q'*Q;
Wc=W(Z1:Z2);

list1=10.^(-12:1:0);
C1=length(list1);
test=zeros(1,C1);
TMP=zeros(C1,Nimage);
W0=W;

Tresh=0.1;
iSp=Sp<Tresh*mean(Sp(Z1:Z2));
ZZ=Z2-Z1+1;
% ZZ=ZZ-sum(iSp(Z1:Z2));
Den=ZZ*dz;
for k=1:C1
    dw=inv(QQ+list1(k)*LL) * (Q'*dR-list1(k)*LL*Wc');
    ratio=dw'./Wc;
    ir=abs(ratio)>TH;
    dw(ir)=TH*sign(ratio(ir)).*Wc(ir);
    dw=[zeros((Z1-1),1); dw; zeros(Nimage-Z2,1)];
    W=W0+dw'; %!!
    W(iSp)=mean(W);
    inx=W>UpperLim ;  W(inx)=UpperLim;
    inx=W<LowerLim ;  W(inx)=LowerLim;
    [test(k) DW2]=testW(W,D1,Sp0,st2,jnx,dz,Z);
    TMP(k,:)=W;
end

if sum(test>0)*sum(test<0)
    % disp('Fine look');
    test2=test.*shiftv1(test,1);
    %  plt2(test,0*test2);
    [V i]=min(test2);
    lam1=list1(i-1);
    lam2=list1(i);
    t1=test(i-1);
    t2=test(i);
    coef=(t1-t2)/(lam1-lam2);
    b=t1-coef*lam1;
    lamX=-b/coef;

    delta=(lam2-lam1)/20;
    list1=lam1:delta:lam2;
    C1=length(list1);
    test=zeros(1,C1);
    TMP=zeros(C1,Nimage);
    for k=1:C1
        dw=inv(QQ+list1(k)*LL) * (Q'*dR-list1(k)*LL*Wc');
        ratio=dw'./Wc;
        ir=abs(ratio)>TH;
        dw(ir)=TH*sign(ratio(ir)).*Wc(ir);
        dw=[zeros((Z1-1),1); dw; zeros(Nimage-Z2,1)];
        W=W0+dw'; %!!
        W(iSp)=mean(W);
        inx=W>UpperLim ;  W(inx)=UpperLim;
        inx=W<LowerLim ;  W(inx)=LowerLim;
        [test(k) DW2]=testW(W,D1,Sp0,st2,jnx,dz,Z);
        TMP(k,:)=W;
    end
    inx=test<0;
    test(inx)=10^10;
    [V i]=min(test);
    tst=test(i)*100; % in %
    W=TMP(i,:);
else
    inx=test<0;
    test(inx)=10^10;
    [V i]=min(abs(test));
    W=TMP(i,:);
    tst=test(i)*100; % in %
    % disp('Out of the rage');
end

if tst>3
    disp('Test>0');
    alfa=.1;
    W0=W;

    for k=1:100;
        Wp=shiftv1(W,+1);
        Wm=shiftv1(W,-1);
        W0=W;
        W(iSp)=(Wp(iSp)+Wm(iSp))/6+W(iSp)*2/3;
        W(~iSp)=(Wp(~iSp)+Wm(~iSp))/8+W(~iSp)*3/4;
        W01=W;
        W=alfa*W01+(1-alfa)*W0;
       [tst DW2]=testW(W,D1,Sp,st2,jnx,dz,Z);
        if (abs(tst)*100<1)|(tst<0); break; end;
    end      
end