function linewidth=LwDiff(spectrum,sweep);

sp =zeroLine(spectrum,0.05);

N=length(sp);
n=N-1;
x=(0:n);
xn=(x-n/2)/n;


M=12000;
n=M-1;
x=(0:n);
xm=(x-n/2)/n;


if M>N
    sp=interp1(xn,sp,xm);
end

sp=[0 diff(sp)];
[mx i]=max(sp);
A=xm(i);


[mx i]=min(sp);
B=xm(i);


linewidth=abs(A-B)*sweep;
