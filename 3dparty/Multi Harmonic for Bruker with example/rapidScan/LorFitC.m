function z=LorFitC(X,vx,vy)
vx=vx+X(4);
A=-1./(1i*X(3)+vx);
B=X(1)*exp(1i*X(2));
z=vy-A*B;
z=abs(z);


% function z=LorC(X,vx)
% A=-1./(1i*X(3)+vx+X(4));
% B=X(1)*exp(1i*X(2));
% z=A*B;
% %z=abs(z);