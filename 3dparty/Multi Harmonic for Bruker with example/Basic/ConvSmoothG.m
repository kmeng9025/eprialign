function res=ConvSmoothG(b,sigma)
%  function res=ConvSmooth(b,f);
%  res=b;f=5;
ss=size(b); s1=ss(1); s2=ss(2);
t=xForInterp(1,s1);
kern=exp(-t.^2/2/sigma.^2);
res=b;
for i=1:s2
tmp=b(:,i);
yyy=myConv(tmp,kern');
res(:,i)=yyy;
end
% plot(res)