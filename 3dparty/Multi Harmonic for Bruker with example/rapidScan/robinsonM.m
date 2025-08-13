function [S Sx]=robinsonM(D,T2,hm,Nh,betta,fi,vm)
         %robinsonM(G,us,G, 1, 1,  grd,MHz);
% T2;   [us]
% vm;   [MHz]
% D      [G]
% hm     [G]; peak-to-peak modulation amplitude
% fi phase correction [ degrees]

%%
gamma=2*pi*2.802; % MHz/G
R2=1/T2;   
n=150;           % number of harmonics
I=sqrt(-1);
wm=2*pi*vm;
%%
a0=gamma*D+I*R2; % [rad/s]
G=gamma*hm/4; G2=G^2;
GN=zeros(n,length(D)*2);
% n
an=[(a0-n*wm) (a0+n*wm)];
A=sqrt(1-4*G2./(an.^2));
B=1+A;
C=an.*B/2;
GN(n,:)=C;

for r=(n-1):(-1):1
    ar=[(a0-r*wm) (a0+r*wm)];
    GN(r,:)=ar-G2./GN(r+1,:);
end
M=length(GN);
GN0=a0-G2*(1./GN(1,1:M/2)+1./GN(1,(M/2+1):M));
Y0=1./GN0*betta/2;
Y=0*GN;
Y(1,:)=-G./GN(1,:).*[Y0 Y0];

for r=2:n
    Y(r,:)=-G./GN(r-1,:).*Y(r-1,:);
end
%fig2(abs(GN),abs(Y)); 
Yminus=Y(:,1:M/2); 
Yplus =Y(:,M/2+1:M); 
% fig(abs([Yminus Yplus]));
S=-(Yplus+Yminus)/pi;
Sx=-I*(Yplus-Yminus)/pi;
% subplot(1,2,1)
% fig([imag(S);imag(Sx)*100]); colorbar;
%S1=imag(S(1:1,:));
fi=fi/180*pi;
e=exp(sqrt(-1)*fi);
S=imag( S(1:Nh,:)*e );
Sx=imag( Sx(1:Nh,:)*e );
