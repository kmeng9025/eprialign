%%
cd z:\CenterMATLAB\3dparty\EPRI_2D_Mark
%%

clear all; clc;
load DATA

%%
tol_pen=5
tol_tikh=4;
Max_harm=100;

method='pen-rose' 
% method='tikh_0'

z=-1:0.1:1;  % 

Nz=length(z);


%%  Polyfit correction
figure(1); clf
for k=1:16
    subplot(4,4,k);
    plot(Rc(:,k));
    axis tight
end
%break

%%  RECONSTRUCTION
SW=hc(end)-hc(1);
TT=zeros(Ng,Nz,Nhc);

for n=1:Ng
    % frequency-domain method to make phantom
    [v A]=fftM(hc,hc);
    w=ifftshift(2*pi*v);                % w= DC ... wmax/w -wmax/2 ... -dw.
    [W,Z] = meshgrid(w,z);
    T=exp(+1i*Z*g(n).*W);  TT(n,:,:)=T; % f-domain shift matrix
end

%%  Reconstruction
PR=fft(Rc',[],2);
x_PH=zeros(Nz,Nhc);

%% Regul operators
v0=ones(1,Nz);
v1=ones(1,Nz-1);
D0=diag(v0,0);
D1=diag(v1,-1)+diag(-v1,+1);
DD1=D1'*D1;
DD0=D0'*D0;

%%

tic
fH=1; % first harmonic
switch method
    case 'pen-rose'
        for m=fH:Max_harm %length(w)
            L=TT(:,:,m);
            b=PR(:,m);
            xx=pinv(L,tol_pen)*b;
            x_PH(:,m)=xx;
        end
    case 'tikh_0'
        for m=fH:Max_harm %length(w)
            L=TT(:,:,m);
            b=PR(:,m);
            LL=L'*L;
            xx=(LL+tol_tikh*DD0)\L'*b;
            x_PH(:,m)=xx;
        end
    case 'tikh_1'
        for m=fH:Max_harm %length(w)
            L=TT(:,:,m);
            b=PR(:,m);
            LL=L'*L;
            xx=(LL+tol_tikh*DD1)\L'*b;
            x_PH(:,m)=xx;
        end
end
toc

%%
figure(1);
subplot(1,1,1);
set(gca,'FontSize',28);
x_Ph=real(ifft(x_PH,[],2));
x_Ph=x_Ph/max(x_Ph(:));
imagesc(hc,z,x_Ph); tag=colorbar;  %colormap gray
set(tag,'FontSize',18);

%%
figure(1); clf;
set(gca,'FontSize',28);
x_Ph=real(ifft(x_PH,[],2));
x_Ph=x_Ph/max(x_Ph(:));
surf(hc,z,x_Ph, 'Linestyle', 'none'); tag=colorbar;  %colormap gray
set(tag,'FontSize',18);
axis tight
