function [image,Siter,err]=iradonMem(Rexp,theta,lamda,Niter,ImageSize)
%      INPUT PARAMETERS
% Rexp - set of experimental projections
% theta - set of angles in grades
% Lamda - straring parameter for iteration 
% Niter - number of iterations;
%      OUTPUT PARAMETERS
% image - reconstructed image
% Siter - (%)  where iteration procedure is auto-stopped 
% MS error between experimental and obtained projections

Nx = 2*floor(size(Rexp,1)/(2*sqrt(2))); % image size
if ImageSize>Nx; N=Nx ; else N=ImageSize; end

GuessImage=iradonC(Rexp,theta,N,'freq');
inx=GuessImage<0; GuessImage(inx)=0;% non-negativity !

xxx=ConvSmooth(GuessImage,20);    % Smoothed Image
xxx=ConvSmooth(xxx',20)';    % Smoothed Image
xxx=xxx+max(max(xxx))*1.5;   % Lifting from zero plane
image=xxx/sum(sum(xxx));     % Image for 1st iteration

Rt=radonD(image,theta); %first guess prjs
Lt=length(Rt(:,1));
Le=length(Rexp(:,1));
T=length(theta);
delta=Lt-Le;

if delta>0
    d2=round(delta/2);
    z1=delta-d2;
    z2=delta-z1;
    zr1=zeros(z1,T);
    zr2=zeros(z2,T);
    Rexp=[zr1' Rexp' zr2']';
end

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
Rt1=Rt;
stopIter=0;
thetaRad=theta*pi/180;

for iter=1:Niter;
    iter/Niter*100
    err0=MSE(Rexp,Rt) ;
    dR=Rt-Rexp; % error in Projections
    dSdF=zeros(N);
    for i=1:length(theta) % Theta loop
        dRi=dR(:,i)';
        p=x.*cos(thetaRad(i))+y.*sin(thetaRad(i));
       
         inx=p>maxP; p(inx)=maxP;
         inx=p<minP; p(inx)=minP;
        pUp=ceil(p);  pDown=floor(p);
        
        %dSdF=dSdF+dRi(pDown+Np2).*(pUp-p)+dRi(pUp+Np2).*(p-pDown);
        %dSdF=dSdF+dRi(pUp+Np2);
        dSdF=dSdF+dRi(round(p)+Np2); % The best result
    end
    
    dSdF=dSdF/N^2;
    dSdF=dSdF/sqrt(sum(sum(dSdF.*dSdF)));
    
    dEdF=entropyDer(image);
    dEdF=dEdF/sqrt(sum(sum(dEdF.*dEdF)));
    
    dImage=(1-lamda*(dSdF-dEdF) );
    inx=dImage<0;  dImage(inx)=1;
    
    imageNew=image.*dImage;

    Rt1=radonD(imageNew,theta); 
    err1=MSE(Rexp,Rt1);
    improv=err1/err0;
    if err1>err0 ;  lamda=lamda/2; stopIter=stopIter+1; 
    else Rt=Rt1; 
         image=imageNew; 
         stopIter=0;
    end;
        
    if stopIter>6;  break; end;
    if isnan(err0); break; end;
end

Siter=round(iter/Niter*100);
err=err1;
