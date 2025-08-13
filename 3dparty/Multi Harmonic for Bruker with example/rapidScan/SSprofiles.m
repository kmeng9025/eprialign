function [fitWidths,LWidths, Spatial, LorImage,freq,trustV]=SSprofiles(Img,freq,deltaH,LowerLim,UpperLim);
% LowerLim=LowerLim/1000;
% UpperLim=UpperLim/1000;
s=size(Img);
N=s(1);
fitWidths=zeros(1,N);
LWidths=fitWidths;
Spatial=zeros(1,N);
LorImage=Img*0;
sw=freq(end)-freq(1);
dh=abs(freq(1)-freq(2));
options = optimset('LargeScale','off','TolFun',0,...
    'LevenbergMarquardt','on');

for j=1:N

    p0=Img(j,:);
    p=p0;

    [pos lwds]=LwX(p,sw); % half widths
    LWidths(j)=lwds/2/1000;
    if LWidths(j)>UpperLim ; LWidths(j)=UpperLim; end;
    if LWidths(j)<LowerLim ; LWidths(j)=LowerLim; end;
    kReal=LWidths(j);

    err0=10^100;
    sumP=sum(p);

    if sumP<=0;
        Coef0=[0 0];
    else
        for k=3:3
            kk=1+(k-3)/20;%
            % z=[(.8+0.4*k/10)*sumP*dh kReal ]
            % z=[(.8+0.4*k/10)*sumP kReal ];
            z=[sumP*dh kReal*kk freq(pos)];
            [Coeff,err(k)]=lsqnonlin('LorFitx',z,[],[],[],freq,p);
            if err(k)<err0;
                Kx=k;
                err0=err(k);
                Coef0=Coeff;
            end
        end
    end
    Coeff=Coef0;
    trust=1*err0/sum(p.^2);

    % Spatial(j)=sum(Lor(Coeff,freq))*dh;
    %subplot(1,2,1);
    dw=Coeff(2);
    if dw>UpperLim ; dw=UpperLim; end;
    if dw<LowerLim ; dw=LowerLim; end;
    if trust>1   ; dw=(LowerLim+UpperLim)/2; end;
    if trust>1   ; Coeff(1)=0; end;
    trustV(j)= trust;
    % Coeff(3)=0;  %%% !!!  to avoid negative point in the image

    Coeff(2)=dw;
    Widths(j)=dw;  % half widths
    
    if Coeff(1)<0; Coeff(1)=0; end ; % % Spatial point
    p=Lorx(Coeff,freq);

    if Coeff(1)<10^(-10);
        fitWidths(j)=LowerLim;
    else
        fitWidths(j)=Lw(p,sw)/2/1000; % half widths
    end

    Spatial(j)=Coeff(1)/dh;
    % plt2(p,Img(j,:));
    LorImage(j,:)=p';
    %     plot(freq,,'k');
     plot(freq,p,'-gx'); hold on;
     plot(freq,p0,'r'); hold off;
    title(num2str(j));
    % subplot(1,2,2);
    %     plot(err)
    % [zk K]
    axis tight

end

fitWidths=fitWidths*2;
LWidths=LWidths*2;

returnStop=0;
