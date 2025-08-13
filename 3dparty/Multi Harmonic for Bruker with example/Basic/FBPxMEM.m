function [FBP,MEM]= FBPxMEM(numSp,lamda,Niter,algorithm,saveName,loadData);
% numSp - number of projections. Two values are 
% avaliable for FBP 30 and 60 
% algorithm  1.-MEM 2.-FBP 0.-both
% Niter number of iterations for MEM
% lamda - mem parameter = 0.5 is good 
% saveName - file name to save result images
% loadData - file name to load projections from
load(loadData,'Rdeco','theta');
% Rdeco=Rdeco';
FBP=0; MEM=0;
%lamda=0.5; Niter=2500  ;
Rdeco=scale2Times(Rdeco);
Rdeco=scale2Times(Rdeco);
Rdeco=scale2Times(Rdeco);
s=size(Rdeco);


%Normalization
Rdeco=normaPlus(Rdeco);

Npoints=s(1);
Nimage=round(Npoints/sqrt(2))
Nang=s(2);


for i=1:s(2)
    p=Rdeco(:,i);
    p=zeroLine(p,0.05);
    Rdeco(:,i)=p;
end

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


switch algorithm
    case 1  %                      1.MEM
        [MEM,Siter,err]=iradonMem2(Rdeco,theta,lamda,Niter,Nimage);
    case 2  %                       2.FBP
        FBP=iradonMiss(Rdeco,theta,176,5,'freq');
        FBP=FBP/sum(sum(FBP));
    otherwise  % both
        [MEM,Siter,err]=iradonMem2(Rdeco,theta,lamda,Niter,Nimage);
        FBP=iradonMiss(Rdeco,theta,176,5,'freq');
        FBP=FBP/sum(sum(FBP));
end


sound(cos(1:1000),10000);
save(saveName);

