function [Sp,tst]=SmoothSpatial(D,re1,LL,D1,W,st1,deltaH,theta,M,Z,dz,jnx);
Nimage=D.Nimage; 
% Sp - spatial profile
% tst - Smoot test in %

Z1=Z(1);
Z2=Z(2);
% LZ=length(Z);
% if LZ>2
%    n=(LZ-2)/2;
%    jnx=ones(Z2-Z1+1,1);
%    for i=1:n
%        iz=[Z(2+i*2-1) Z(2+2*i)]-Z(1);
%         jnx(iz(1):iz(2))=0;
%    end
% end

dRm=LorRadon(W,deltaH,theta,M,[Z1 Z2]);
RR=dRm'*dRm;
f=dRm'*re1;

list1=10.^(-5:1:5);
C1=length(list1);
test=zeros(1,C1);
TMP=zeros(C1,Nimage);
for k=1:C1
    F=list1(k)*LL+RR;
    F=inv(F);
    Sp=F*f;
    Sp=Sp.*jnx; % !!!!
    in=Sp<0;  Sp(in)=0;
     Sp=[zeros(1,(Z1-1)) Sp' zeros(1,Nimage-Z2)];
    [test(k) tmp]= testW(Sp,D1,ones(1,length(Sp)),st1,jnx,dz,Z);
        % test(k)=(sum(abs(D1*Sp))/(length(Sp)*dz)-st1)/st1;
   
    TMP(k,:)=Sp;
end

if sum(test>0)*sum(test<0)

    test2=test.*shiftv1(test,1);
    %  plt2(test,0*test2);
    [V i]=min(test2);

    lam1=list1(i-1);
    lam2=list1(i);
%     t1=test(i-1);
%     t2=test(i);
%     coef=(t1-t2)/(lam1-lam2);
%     b=t1-coef*lam1;
%     lamX=-b/coef;
    delta=(lam2-lam1)/20;
    
    % list1=(lamX-4*delta):delta:(lamX+4*delta);
    list1=(lam1-delta):delta:lam2;
    C1=length(list1);
    test=zeros(1,C1);
    TMP=zeros(C1,Nimage);
    for k=1:C1
        F=list1(k)*LL+RR;
        F=inv(F);
        Sp=F*f;
        Sp=Sp.*jnx; % !!!!
        in=Sp<0;  Sp(in)=0;
        Sp=[zeros(1,(Z1-1)) Sp' zeros(1,Nimage-Z2)];
        [test(k) tmp]= testW(Sp,D1,ones(1,length(Sp)),st1,jnx,dz,Z);
        TMP(k,:)=Sp;
    end
   %  j=test<0; test(j)=10^10;
    [V i]=min(abs(test));
    tst=test(i); % in %
    Sp=TMP(i,:);
else
   % j=test<0; test(j)=10^10;
    [V i]=min(abs(test));
    Sp=TMP(i,:);
    tst=test(i); % in %
    disp('Out of the rage');
end