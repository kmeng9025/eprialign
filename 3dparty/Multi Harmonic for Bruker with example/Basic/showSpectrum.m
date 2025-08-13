% function showSpectrum


g=0; M=2000; b1=7.5; R=2.35; G=180;
 sss=G*0.02;
 
 
[dX,dP]=spectrum_1_line(g,b1,M,R,sss);
%K=300:400;
plot(dX,dP/max(dP));

AB=-max(dP)/min(dP)
 hold on;
% 
% R=2.2; G=100;  sss=G*0.02;
% [dX1,dP1]=spectrum_1_line(g,b1,M,R,sss);
% plot(dX1,dP1/max(dP1),'Color','red');
% 
% hold off;

x=-5:.01:5;
psi=G*0.02/R;
[xres,y1]=dysonxRsur(x,R,psi);
plot(-x,y1/max(y1),'-r');
AB2=-max(y1)/min(y1)
hold off;