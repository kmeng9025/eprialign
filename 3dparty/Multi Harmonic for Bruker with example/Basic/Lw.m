function linewidth=Lw(spectrum,sweep)

%sp =zeroLine(spectrum,0.05);
sp=spectrum;
N=length(sp);
n=N-1;
x=(0:n);
xn=(x-n/2)/n;


M=2000;
if M>N
    n=M-1;
    x=(0:n);
    xm=(x-n/2)/n;
    sp=interp1(xn,sp,xm);
else
    M=N;
end

mx=max(sp);
inx=sp>mx/2;

N=length(sp);
linewidth=round(10000*sum(inx)/N*sweep)/10;
dh=sweep/M;
h=xForInterp(sweep,M);
plot(h,inx,h,sp/max(sp)); 
axis tight
title(['Linewidth=' num2str(linewidth) 'mG']); 