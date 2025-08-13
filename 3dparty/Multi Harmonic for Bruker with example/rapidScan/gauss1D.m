function res=gauss1D(N,T);
%T=1;
% N=1000;

x=xForInterp(N,N)
x2=x.^2;
T2=T^2;
A=x2/T2/2
res=exp(-A);
res=res/sum(res);

% plot(res,'-x')
% 
% x1=(X-a); x2=x1.^2*1/T^2; 
% y1=(Y-b); y2=y1.^2*1/T^2; 
% z3=x2+y2;
% res=exp(-z3);
% %mesh(X,Y,exp(-z3));

