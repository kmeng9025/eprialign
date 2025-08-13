% conv[data_,kern_]:=Module[{shft,tmp3,xxx,tmp2,tmp,L},
%    L=Length[data];
%    nnn=Round[FirstMom[data]];
%    tmp=Fourier[ RotateLeft[data,nnn] ];
%    tmp2=Fourier[ RotateLeft[kern,L/2-1] ];
%    tmp3=InverseFourier[tmp tmp2 Sqrt[L]];
%    xxx=RotateRight[tmp3,nnn]
function res=myConv(data,kern)

   L=length(data);
   % nnn=Round[FirstMom[data]];
   tmp=fft(data);
   tmp2=fft(fftshift(kern));
   tmp3=real(ifft(tmp.*tmp2* sqrt(L)));
   res=tmp3;

