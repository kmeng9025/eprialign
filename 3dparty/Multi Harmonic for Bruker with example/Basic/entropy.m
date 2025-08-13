function res=entropy(Image);

S=sum(sum(Image));
if S~=0
Image=Image/S;
inx=Image==0;
Image(inx)=1;%  Log(0)=- infinity
res=Image.*log2(Image); % !!!???
res=-sum(sum(res));
else
    res=1
end