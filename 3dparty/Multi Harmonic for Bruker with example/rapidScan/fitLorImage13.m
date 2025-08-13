function [Sp,W,test1,test2,S]=fitLorImage13(D,CutOffTheta,averageW,h,theta,Rexp,Niter,STest,Z,Acc)
global Nimage UpperLim LowerLim Lmodel deltaH
Nimage=D.Nimage; Lmodel=D.Lmodel;
LowerLim=D.limits(1);  UpperLim=D.limits(2); deltaH=D.deltaH;
%lamda - pair of start lamdas

Z1=Z(1);  Z2=Z(2);  N=Z2-Z1+1;
O1=ones(1,N-1);
D0=diag(ones(1,N),0);
D1=diag(O1,-1)-D0;
D2=( diag(O1,-1)-D0+diag(O1,1) )/2;
order=3;
if order==1;  LL=D1'*D1; end;
if order==2;  LL=D2'*D2; end;
if order==3;  LL=(D2'*D2+D1'*D1)/2; end;

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
list0=10.^((-C:C)/C );
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

st1=0.12;
st2=2;
Niter1=12;
TH=.1;
TH2=10;
Accuracy=Acc; % in percent
CutOffTheta =70;


tic
Aim=0;
Waim=0*W;

disp('---Start----')
Err0=Err;
for i=1:4
    [Sp test1]=MinSpatial(D,re1,LL,D1,W,st1,deltaH,theta(inx1),M,Z,dz);
    [W,test2]=MinWidths(D,re2,LL,D1,Sp,W,st2,deltaH,theta(inx2),M,Z,dz,TH);
end
for www=1:10
    [Sp test1]=MinSpatial(D,re1,LL,D1,W,st1,deltaH,theta(inx1),M,Z,dz);
    [W,test2]=MinWidths(D,re2,LL,D1,Sp,W,st2,deltaH,theta(inx2),M,Z,dz,TH);
    % Error test
    P=LorRadonSp(Sp,W,deltaH,theta(inx0),M,[Z1 Z2]);
    R=sum(P')';
    dR=re-R;
    Err=sum(dR.^2);
    if abs((Err-Err0)/Err0)>0.05;
        Err0=Err; Sp0=Sp; W0=W;
    else disp('break'); break;
    end;
    Err
end
Sp=Sp0; W=W0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S=100*sign(test1)+ sign(test2)
S=101
switch S
    case -101
        disp('Overestimated effective gradients');
    case -99
        disp('Overestimated R-gradient; Underestimated O-gradient');
        [W,test2]=SmoothWidths(D,re2,LL,D1,Sp,W,st2,deltaH,theta(inx2),M,Z,dz,TH);
    case 99
        disp('Overestimated O-gradient; Underestimated R-gradient;');
        [Sp test1]=SmoothSpatial(D,re1,LL,D1,W,st1,deltaH,theta(inx1),M,Z,dz);
    case 101
        disp(' loop ');
end
Sp0=Sp; W0=W;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if S==-100000;
    for i=1:Niter
        [W,test2]=SmoothWidths(D,re2,LL,D1,Sp,W,st2,deltaH,theta(inx2),M,Z,dz,TH);
        [Sp test1]=SmoothSpatial(D,re1,LL,D1,W,st1,deltaH,theta(inx1),M,Z,dz);

        P=LorRadonSp(Sp,W,deltaH,theta(inx0),M,[Z1 Z2]);
        R=sum(P')';
        dR=re-R;
        Err=sum(dR.^2);
        if abs((Err-Err0)/Err0)>0.05;
            Err0=Err; Sp0=Sp; W0=W;
        else disp('break'); break;
        end
    end
end
Sp=Sp0; W=W0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test1=round(test1*100)/100;
test2=round(test2*100)/100;

