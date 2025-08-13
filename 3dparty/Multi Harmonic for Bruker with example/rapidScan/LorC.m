function z=LorC(X,vx)
% 1 amplitude
% 2 phase
% 3 width
% 4 offset
A=-1./(1i*X(3)+vx+X(4));
B=X(1)*exp(1i*X(2));
z=A*B;
%z=abs(z);