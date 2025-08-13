function showSSprofiles(fn,FBP,fitW,W,Sfbp,Spatial,Widths,x,txt,ZM1,ZM2)

Spatial=Spatial(ZM1:ZM2);
Sfbp=Sfbp(ZM1:ZM2);
W=W(ZM1:ZM2);
fitW=fitW(ZM1:ZM2);
FBP=FBP(ZM1:ZM2,:);
x=x(ZM1:ZM2);

subplot(3,1,1);
% plot(xi2fbp);
imagesc(FBP); % axis ([50 201-50 30 200-30 ])
title(txt)

subplot(3,1,2);
plot(x,fitW);   hold on;
plot(x,W,'-r');   
plot(x,Widths(ZM1:ZM2),'-g'); hold off; axis tight;


title 'Blue -with fitting, Red- without fitting   only for FBP'
subplot(3,1,3);
plot(x,Sfbp,x,Spatial);   axis tight;
title 'Blue - from image, Green is expected Spatial profile'

saveas(gcf,fn,'emf');

% FBP=xFBP;