function [t,Mx,My]=RsSteadyState(T2,hm,Vm,dH,BW)% progonka y''-y=0; y(0)=1;
%T2 s
%hm - mudulation amplitude, G
%Vm - modulation frequency, Hz
%dH - magnetic field offset, G
% BW - signal bandwidth, Hz

N=round(BW/Vm);
M=4096*4;
gamma=1.76e7;
y0=hm*gamma*T2/2;
lamda=2*pi*Vm*T2; 
dx=dH*gamma*T2;
y=sqrt(-1)/2*y0;


I=sqrt(-1); % +/- ?
a=zeros(1,2*N+1); % diagonal
f=zeros(1,2*N+1); % right constants
b=zeros(1,2*N); % up
c=zeros(1,2*N); %down



x=1+dx*I;
k=N:(-1):-N;
a=lamda*k*I+x;
b(:)=y;
c(:)=y;
f(N+1)=I; % +/- ?


res=progonka3(a,b,c,f);
clear a b c f y x
a=res(1:N);           % alfa
b=res( (N+2):(2*N+1) ); % betta
a=fliplr(a);
a0=res(N+1);

MM=M-1;
tau=(0:MM)/MM*1/lamda*2*pi;
mtx=zeros(N,1000);

sum=a0;

for k=1:N
    sum=sum+a(k)*exp(I*k*lamda*tau)+...;
            b(k)*exp(-I*k*lamda*tau);
end

%cw=imag( a(1)+b(1) );
% cw=real( a(1)-b(1) );

% mtx=conj(mtx);
Mx=T2*real(sum);
My=T2*imag(sum);

t=(0:M)/M*2*pi/lamda*T2;
t=t(1:(end-1));

