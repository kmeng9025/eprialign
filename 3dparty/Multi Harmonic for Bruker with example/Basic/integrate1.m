function res=integrate1(sp);
% function res=integrate1(sp);

ss=size(sp);
s=0;
res=zeros(ss);

if ss(1)>ss(2);
    le=ss(1);
else
    le=ss(2);
end

for i=1:le
    s=s+sp(i);
    res(i)=s;
end