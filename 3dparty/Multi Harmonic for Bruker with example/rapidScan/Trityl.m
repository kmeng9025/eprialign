function y=TritylC(X,vx)
vx=vx+X(4);
A=-1./(1i*X(3)+vx);
B=X(1)*exp(1i*X(2));

%%
F=0.15;
broad=14.5e-3;
x=vx/broad;
xx=x.^2;
envelop1=F./(1+xx)+(1-F)*exp(-0.693*xx);

HP=0.166;
x=(vx-HP/2)/broad;
xx=x.^2;
envelop2=F./(1+xx)+(1-F)*exp(-0.693*xx);

x=(vx+HP/2)/broad;
xx=x.^2;
envelop3=F./(1+xx)+(1-F)*exp(-0.693*xx);

envelop=envelop1+(envelop2+envelop3)*.033;
N=length(envelop);
y=conv(A*B,envelop);

%y=conv(envelop,envelop);
y=y(N/2:N/2+N-1);

%z=vy-y;
%z=abs(z);

% function z=LorC(X,vx)
% A=-1./(1i*X(3)+vx+X(4));
% B=X(1)*exp(1i*X(2));
% z=A*B;
% %z=abs(z);