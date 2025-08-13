function res=lorenz1(len,dx)
x=(1:len); 
y=1./((x-len/2).^2+ dx^2)/pi;
y=dx*y;
y=y/sum(y);
res=y;

%plot(y);
