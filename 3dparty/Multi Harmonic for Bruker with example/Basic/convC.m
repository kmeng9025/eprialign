function res=convC(data,kern)

   L=length(data);
   ss=size(data);
   
   
   
   tmp=fft(data);
   tmp2=fft(fftshift(kern));
   tmp3=real(ifft(tmp.*tmp2));
   res=tmp3;
   


   
% 
% x=zeros(ss);
% 
% I=sqrt(-1);
% xxx= 2*ifft(I*sp);
% 
% L=length(xxx);
% 
% xxx(1:round(L/2))=0;
% res=fft(xxx);
% 
