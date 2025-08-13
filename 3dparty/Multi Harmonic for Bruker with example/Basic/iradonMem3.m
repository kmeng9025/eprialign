function [image,t,error,TEST]=iradonMem3(Rexp,theta,lamda,Niter,d,Nimage)
%      INPUT PARAMETERS
% Rexp - set of experimental projections
% theta - set of angles in grades
% Lamda - straring parameter for iteration 
% Niter - maximum number of iterations
%      OUTPUT PARAMETERS
% image - reconstructed image
% Siter - (%)  where iteration procedure is auto-stopped 
% MS error between experimental and obtained projections
% filter parameter for iradon, better d=0.01 -> 1

% s=size(Rexp);
% for i=1:s(2)
%     p=Rexp(:,i);   
%     p=p/sum(p);
%     Rexp(:,i)=p;
% end

%Niter=2000;
% Nx = 2*floor(size(Rexp,1)/(2*sqrt(2))); % image size
% if ImageSize>Nx; N=Nx ; else N=ImageSize; end

FBP=iradon(Rexp,theta,'linear','Hamming',1,Nimage);
inx=FBP<0; FBP(inx)=0;% non-negativity !
FBP=FBP/sum(sum(FBP));  

s=size(FBP); N=s(1); % size of Image

GuessImage=iradon(Rexp,theta,'linear','Hamming',1,Nimage);
% GuessImage=iradon(Rexp,theta,'linear','Hamming',d,N);
inx=GuessImage<0; GuessImage(inx)=0;% non-negativity !

% xxx=ones(N);xxx(1,1)=1.1;
xxx=GuessImage;
 xxx=ConvSmooth(GuessImage,d);    % Smoothed Image
 xxx=ConvSmooth(xxx',d)';    % Smoothed Image
% xxx=xxx+max(max(xxx))*110;   % Lifting from zero plane
image=xxx/sum(sum(xxx));     % Image for 1st iteration
clear xxx GuessImage  % Clear memory

Lexp=length(Rexp(:,1));
Rt=radon(image,theta,Lexp); %first guess prjs
Rfbp=radon(FBP,theta,Lexp); %first guess prjs

s=size(Rexp); 
Np=s(1);
Np2=ceil(Np/2);
ip=(1:Np)-Np2;
minP=min(ip);
maxP=max(ip);

xLine = (1:N)-ceil(N/2);
x = repmat(xLine, N, 1);% X matrix    
y = rot90(x);           % Y matrix  

err1=0;
xi2=0;

Rt1=Rt;

STDiv=std(Rexp(1:30,:));

stopIter=0;
thetaRad=theta*pi/180;

dR=Rfbp-Rexp; % error in Projections
S1=sum(dR.^2);
xi2=sum(S1./(STDiv.^2));
errorFBP=xi2/(s(1)*s(2));%% ERROR before iteration
error=errorFBP;
tic
for iter=1:Niter;
    t=iter/Niter*100; % time in %
    dR=Rt-Rexp; % error in Projections
  
    %% ERROR before iteration
    S1=sum(dR.^2);
    xi2=sum(S1./(STDiv.^2));
    err0=xi2/(s(1)*s(2));

    dSdF=zeros(N);
    for i=1:length(theta) % Theta loop
        dRi=dR(:,i)';
        p=x.*cos(thetaRad(i))+y.*sin(thetaRad(i));
        inx=p>maxP; p(inx)=maxP;
        inx=p<minP; p(inx)=minP;
        dSdF=dSdF+dRi(round(p)+Np2); % The best result
    end
    
    dSdF=dSdF/N^2;
    dSdF=dSdF/sqrt(sum(sum(dSdF.*dSdF)));
    
    dEdF=entropyDer(image);
    dEdF=dEdF/sqrt(sum(sum(dEdF.*dEdF)));
    
    dImage=(1-lamda*(dSdF-dEdF) );
    inx=dImage<0;  dImage(inx)=1;
    imageNew=image.*dImage;    
    Rt1=radon(imageNew,theta,Lexp);     
    dR1=Rt1-Rexp; % error in Projections
  
    
    %% ERROR after iteration
    S1=sum(dR1.^2);
    xi2=sum(S1./(STDiv.^2));
    err1=xi2/(s(1)*s(2));
    
    improv=err1/err0;
    rE=err1/err0-1;
    if rE>0 ;  lamda=lamda/2; stopIter=stopIter+1; 
    else Rt=Rt1; 
        image=imageNew; 
        stopIter=0;
    end;
    TEST=0.5*norm(dSdF(:)-dEdF(:))^2
    
    if (stopIter>7)&(TEST<.1); disp('stop Iter>6'); break; end;
    if isnan(err0); disp('isnan error');break; end;
    if xi2<=s(1)*s(2); disp('xi2 criterium ');break;end;
    if abs(rE)<0.005; disp('slow convergence'); break; end;
    
    disp([num2str(t) '% done, err=' num2str(err1) ', FBP err=' num2str(errorFBP)  ', lambda=' num2str(lamda)] );
    error=[error err1];
%     toc
end
save MEMpar t error lamda Niter TEST
Siter=round(iter/Niter*100);
err=err1;

image=image';
toc