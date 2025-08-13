function res=gauss2D(a,b,N,T);
% res=gauss2D(a,b,N,T);
% N=100;
% a=N/2;
% b=N/2;
% T=10;

[X,Y] = meshgrid(1:N,1:N);

%Z=X.*Y;

x1=(X-a); x2=x1.^2*1/T^2; 
y1=(Y-b); y2=y1.^2*1/T^2; 
z3=x2+y2;
res=exp(-z3);
%mesh(X,Y,exp(-z3));

