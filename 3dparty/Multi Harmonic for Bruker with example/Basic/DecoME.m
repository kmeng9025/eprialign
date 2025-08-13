function  res=DecoMe(kern,out,epsilon,tol);
% out - convoluted spectrum
% kern - point spread function
% tol - tolirance // for pseudo -inv- matrix
% epsilon  // regularization parameter

len=length(kern);
K=zeros(len);

for i=1:len
    rv=rotatev(kern,i-1);
    rv=fftshift(rv);
    K(:,i)=fliplr(rv)';
    % K(:,i)=rv';
end
M=4;
M2=M/2;

for k=1:M
    tolx=tol*2^(M2-k);
    Kplus=pinv(K,tolx);
    x=Kplus*out;   inx=x<epsilon; x(inx)=epsilon;
    r=log(x);
    lamb=-Kplus'*r;
    Kplus=pinv(K,2*tolx);
    xxx=K'*lamb;
    res=exp(-xxx);
    delta=(K*res-out);
    err(k)=delta'*delta;
end

plot2(K*res,out); 
[value,k]=min(err);

tolx=tol*2^(M2-k);
Kplus=pinv(K,tolx);
x=Kplus*out;   inx=x<epsilon; x(inx)=epsilon;
r=log(x);
lamb=-Kplus'*r;
Kplus=pinv(K,2*tolx);
xxx=K'*lamb;
res=exp(-xxx);

