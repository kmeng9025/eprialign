function tmp2Dx=shiftV(data2D,grad,koef)
% load par;
ss=size(data2D);
n=ss(2);
% xxx=1:14; yyy=[fliplr(xxx) 0 xxx]*koef;

sgn=ones(1,n); for i=1:(round(n/2)-1) sgn(i)=-1; end;

grad=abs((1:n)-n/2); %%% !!!

grad=grad.*sgn% .*sgn;

yyy =grad/max(grad)*koef;
tmp2Dx=data2D;

for a=1:n
    clear tmp tmp1 tmp2
    tmp=data2D(:,a); 
    tmp1=yyy(a);
    tmp2=rotatev(tmp',round(tmp1) )';
    tmp2Dx(:,a)=tmp2;
end;

