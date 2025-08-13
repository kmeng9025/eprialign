function res=hevi(x);

inx=x==0;
res=(sign(x)+1)/2;
res(inx)=1;