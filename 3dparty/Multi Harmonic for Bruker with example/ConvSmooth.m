function res=ConvSmooth(b,f);
%  function res=ConvSmooth(b,f);
%  res=b;f=5;
ss=size(b); s1=ss(1); s2=ss(2);
kern=lorenz1(s1,f);
res=b;
for i=1:s2
tmp=b(:,i);
yyy=myConv(tmp,kern',1);
res(:,i)=yyy;
end
% plot(res)