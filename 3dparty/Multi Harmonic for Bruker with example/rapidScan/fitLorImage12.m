function [Sp,W,test1,test2,st2]=fitLorImage12(D,CutOffTheta,averageW,h,theta,Rexp,Niter,STest,Z,Acc)
% global Nimage UpperLim LowerLim Lmodel deltaH
Nimage=D.Nimage; Lmodel=D.Lmodel;
LowerLim=D.limits(1);  UpperLim=D.limits(2); deltaH=D.deltaH;
% lamda - pair of start lamdas

Z1=Z(1);
Z2=Z(2);
N=Z2-Z1+1;
LZ=length(Z);
if LZ>2
   n=(LZ-2)/2;
   jnx=ones(Z2-Z1+1,1);
   for i=1:n
       iz=[Z(2+i*2-1) Z(2+2*i)]-Z(1);
        jnx(iz(1)-1:iz(2)+1)=0;
   end
end


O1=ones(1,N-1);
D0=diag(ones(1,N),0);
D1=diag(O1,-1)-D0;
D2=diag(O1,-1)-2*D0+diag(O1,1);

Dsp=(D2+D2)/2; LLsp=Dsp'*Dsp;
Dw=D1;  LLw=Dw'*Dw;
clear D1 D2
% order=2;
% 
% if order==1;  LL=D1'*D1; end;
% if order==2; 
%     LL=D2'*D2; 
%     D1=D2;
% end;
% if order==3;  
%     LL=(D2'*D2+D1'*D1)/2; 
%     D1=(D1+D2)/2
% end;

s=size(Rexp);
M=s(1);                   % number points in one projection
Nproj=s(2);               % number of projections
Re=Rexp(:);               % experimental data

W=ones(1,Nimage)*averageW;     % first guess of linewidths

inx1=theta<-CutOffTheta; % CutOffTheta;
tmp=repmat(inx1,M,1);
itmp=tmp(:); % logical


dz=Lmodel/Nimage;
re1= Re(itmp);

inx0=(theta<0); % CutOffTheta;
tmp=repmat(inx0,M,1);
itmp=tmp(:); % logical
re=Re(itmp);


inx2=(theta<0); % CutOffTheta;
tmp=repmat(inx2,M,1);
itmp=tmp(:); % logical
re2=Re(itmp);


C=5;
C2=C/2;
% list0=10.^((-C:C)/C );
list=10.^( (-C2:C2)/C2 );
list0=list;

C=length(list);


st1=STest(1);
st2=STest(2);
GlobErr=10^10;
Sp=ones(1,Nimage);

Test=10^100;
Err=10^100;
test1=1000;
test2=1000;
count=0;

% st1=0.12;
% st2=2;
Niter1=5;
TH=0.05;
TH2=10;
Accuracy=Acc; % in percent
CutOffTheta =70;


tic
Aim=0;
Waim=0*W;

disp('- Step1; To Minimum Error -')
Err0=Err;
[Sp test1]=MinSpatial(D,re1,LLsp,Dsp,W,st1,deltaH,theta(inx1),M,Z,dz);
[W,test2]=MinWidths(D,re2,LLw,Dw,Sp,W,st2,deltaH,theta(inx2),M,Z,dz,4*TH,jnx);
[W,test2]=MinWidths(D,re2,LLw,Dw,Sp,W,st2,deltaH,theta(inx2),M,Z,dz,3*TH,jnx);
[W,test2]=MinWidths(D,re2,LLw,Dw,Sp,W,st2,deltaH,theta(inx2),M,Z,dz,2*TH,jnx);

for www=1:10
    [Sp test1]=MinSpatial(D,re1,LLsp,Dsp,W,st1,deltaH,theta(inx1),M,Z,dz);
    [W,test2]=MinWidths(D,re2,LLw,Dw,Sp,W,st2,deltaH,theta(inx2),M,Z,dz,2*TH,jnx);
    % Error test
    P=LorRadonSp(Sp,W,deltaH,theta(inx0),M,[Z1 Z2]);
    R=sum(P')';
    dR=re-R;
    Err=sum(dR.^2);
    if abs((Err-Err0)/Err0)>0.01;
        % abs((Err-Err0)/Err0)*100
        Err0=Err; Sp0=Sp; W0=W;
    else disp('break'); break;
    end;
end
Sp=Sp0; 
W=W0;
[test2x0 tmp]= testW(W0,Dw,Sp0,st2,jnx,dz,Z);
[test1x0 tmp]= testW(Sp,Dsp,ones(1,length(Sp)),st1,jnx,dz,Z);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if test2x0>0;
    [W test2]=SmoothMap(W0,Dw,st2,Sp0,dz,Z,jnx,3);
    if test2>Accuracy(2)/100;
        [W test2]=SmoothMap(W,Dw,st2,Sp0,dz,Z,jnx,0);
    end
    SmTest=1;
    % plot(Z1:Z2,W(Z1:Z2))
else
    [test2 st2]=testW(W,Dw,Sp0,st2,jnx,dz,Z);
    SmTest=0; % too noisy
end;


if abs(test1x0)>Accuracy(1)/100;
    [Sp test1]=SmoothMap(Sp,Dsp,st1,ones(1,length(Sp)),dz,Z,jnx,3);
    P=LorRadonSp(Sp,W,deltaH,theta(inx0),M,[Z1 Z2]);
    R=sum(P')';
    dR=re-R;
    Err0=sum(dR.^2);
end

%     [Sp test1]=SmoothSpatial(D,re1,LLsp,Dsp,W,st1,deltaH,theta(inx1),M,Z,dz,jnx);
P=LorRadonSp(Sp,W,deltaH,theta(inx0),M,[Z1 Z2]);
R=sum(P')';
dR=re-R;
Err0=sum(dR.^2)
% ------------------------------------

for iii=1:Niter1
    
   [W0,test2x]=SmoothWidths(D,re2,LLw,Dw,Sp,Sp0,W,st2,deltaH,theta(inx2),M,Z,dz,TH,jnx);
   [Sp0 test1x]=SmoothSpatial(D,re1,LLsp,Dsp,W,st1,deltaH,theta(inx1),M,Z,dz,jnx);
 
    % Error
    P=LorRadonSp(Sp0,W0,deltaH,theta(inx0),M,[Z1 Z2]);
    R=sum(P')';
    dR=re-R;
    Err=sum(dR.^2);

    % Smooth Test
    SmoothTest=(abs(test1x)>Accuracy(1))| (abs(test2x)>Accuracy(2));
    ErrorTest=Err>Err0;
    
    if SmoothTest|ErrorTest; 
        break 
    else
        W=W0;
        Sp=Sp0;
        Err0=Err;
        disp([txnum('N_iter=',iii) txnum('; test1=',test1) txnum('% ; test2=',test2) txnum('% ; ERR=',1000*Err)]  );
        test1=test1x;
        test2=test2x;
    end;
    % Aim=Aim+SmoothTest;
    % Show current results
   %  disp([txnum('N_iter=',iii) txnum('; test1=',test1) txnum('% ; test2=',test2) txnum('% ; ERR=',1000*Err)]  );

    % Break tests
%     if ~SmoothTest; Waim=W; tst2=test2; tst1=test1; end;
%     if ((Err0-Err)/Err0<0.01)& Aim;  disp('Break with Success ');break; end;
end
% if Aim; W=Waim; test2=tst2; test1=tst1; end;
% time1=toc;
% disp(txnum('! Finish with total time = ',time1));
% % plt3(Widths,W,Sp); axis tight;

test1=round(test1*100)/100;
test2=round(test2*100)/100;










