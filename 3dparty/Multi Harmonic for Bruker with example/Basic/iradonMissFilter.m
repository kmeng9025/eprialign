function I=iradonMissFilter(Re,thetaM,Nimage,iter,interp,filter);
% Missing Anglge Algorithm
% Re - set of prjs
% thetaM - set og angles
% Nimage - reconstructed image size
% iter - number of iterations
% domain -'freq' or 'time'

s=size(Re);
Npoints=s(1);
Nang=length(thetaM);
da=abs(thetaM(1)-thetaM(2)); % angle increment
Ntheta=180/da;        % Number of full set of prjs 
theta=(da/2:da:180)-90;     % Full set of angles
miss=round(Ntheta-Nang);     % number of missing prjs
mi2=miss/2;

pm=zeros(Npoints,mi2);
R=[pm Re pm];

for i=1:iter;
I=iradon(R,theta,interp,filter,1,Nimage);
%iradon(R,theta,interp,filter,frequency_scaling,output_size)
inx=I<0; I(inx)=0;

p1=radon(I,theta(1:mi2));
p2=radon(I,theta(end-mi2+1:end));

delta=length(p1)-Npoints;
if delta>0;
    k=round(delta/2);
    pp1=p1(k:(end-(delta-k)-1),:);
    p1=pp1;
    pp2=p2(k:(end-(delta-k)-1),:);
    p2=pp2;
%else
end
R(:,1:mi2)=p1;
R(:,end-mi2+1:end)=p2;
end

inx=I<0; I(inx)=0;

%I=scale2Times(I);
%I=scale2Times(I')';
% I=scale2Times(I);
% I=scale2Times(I')';
% mesh(I); axis tight 
%mesh(Img)
% I=SandraRadon(Img,theta);
% 
% 
% 
% 
% 
% R=radonD(p,theta);

