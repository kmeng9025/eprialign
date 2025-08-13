function R=normaPlus(R);
%Normalization 
% non-negativity
s=size(R);
for i=1:s(2)
    p=R(:,i);
    p=zeroLine(p,0.05);
    p=p/sum(p);
    inx=p<0;
    p(inx)=0;
    R(:,i)=p;
end
