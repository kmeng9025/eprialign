function [FBP,MEM]= FBPxMEM3(numSp,lamda,Niter,algorithm,saveName,loadData,d);
% numSp - number of projections. Two values are
% avaliable for FBP 30 and 60
% algorithm  1.-MEM 2.-FBP 0.-both
% Niter maximum number of iterations for MEM
% lamda - mem parameter = 0.5 is good
% saveName - file name to save result images
% loadData - file name to load projections from
load(loadData,'Rdeco','theta');
load addData Ra add

% Rdeco=Rdeco';
FBP=0; MEM=0;
%lamda=0.5; Niter=2500  ;
% Rdeco=scale2Times(Rdeco);
% Rdeco=scale2Times(Rdeco);
% Rdeco=scale2Times(Rdeco);

s=size(Rdeco);
%Normalization
Rdeco=normaPlus(Rdeco);
% Npoints=s(1);
% Nang=s(2);
s1=s(1); s2=s1/2;

if round(s1/2)==s1/2
    Rdeco1=zeros(s(1)+1,s(2));
    
    j=1:s1;
    x1=-s2:1:s2;
    x0=-s2+(j-1)/(s1-1)*s1
    
    for i=1:s(2)
        p=Rdeco(:,i);
        p=zeroLine(p,0.05);
        p=interp1(x0',p,x1');
        Rdeco1(:,i)=p;
    end
    Rdeco=Rdeco1;
    clear Rdeco1;
else
    for i=1:s(2)
        p=Rdeco(:,i);
        p=zeroLine(p,0.05);
        Rdeco(:,i)=p;
    end
end
Rdeco=normaPlus(Rdeco);

FBP=iradon(Rdeco,theta,'Ram-Lak');
tmp=size(FBP);
Nimage=tmp(1); FBP=0;

s=size(Rdeco);

switch numSp
    case 30
        % 30 projections _______________________________
        Nproj=32; deltaTheta=180/Nproj;
        theta32=(deltaTheta/2:deltaTheta:180)-90;
        theta32=theta32(2:end-1);
        Rnew=zeros(s(1),30);
        for i=1:s(1)
            tmp=Rdeco(i,:);
            tmp2=interp1(theta,tmp,theta32);
            Rnew(i,:)=tmp2;
        end
        Rdeco=Rnew;
        theta=theta32;
    case 60
        % 60 projections _________________________________
        Nproj=64; deltaTheta=180/Nproj;
        theta64=(deltaTheta/2:deltaTheta:180)-90;
        theta64=theta64(3:end-2);
        Rnew=zeros(s(1),60);
        for i=1:s(1)
            tmp=Rdeco(i,:);
            tmp2=interp1(theta,tmp,theta64);
            Rnew(i,:)=tmp2;
        end
        Rdeco=Rnew;
        theta=theta64;
end


clear tmp tmp2 Rnew
switch algorithm
    case 1  %                      1.MEM
        [MEM,StopT,error]=iradonMem3(Rdeco-Ra,theta,lamda,Niter,d);
        K=1-sum(sum(add));
        MEM=MEM/sum(sum(MEM))*K+add;
    case 2  %                       2.FBP
        FBP=iradonMiss(Rdeco,theta,Nimage,5);
        FBP=FBP/sum(sum(FBP));
    otherwise  % both
        [MEM,StopT,error]=iradonMem3(Rdeco-Ra,theta,lamda,Niter,d,Nimage);
        s=size(MEM);  Nimage=s(1);
        FBP=iradonMiss(Rdeco,theta,Nimage,5);
        % FBP=FBP/sum(sum(FBP));
        K=1-sum(sum(add));
        MEM=MEM/sum(sum(MEM))*K+add;
end


% sound(cos(1:1000),10000);
save(saveName);

