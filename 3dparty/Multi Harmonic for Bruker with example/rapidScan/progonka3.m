function res=progonka3(a,b,c,f);

N=length(a);

% a=zeros(1,N); % diagonal
% f=zeros(1,N); % right constants
% b=zeros(1,N-1); % up
% c=zeros(1,N-1); %down

i=1;
b(1)=b(1)/ a(1);
f(1)=f(1)/ a(1);
a(1)=1;

for i=2:(N-1)
    
    a(i)=a(i)-b(i-1)*c(i-1);
    f(i)=f(i)-f(i-1)*c(i-1);    

    b(i)=b(i)/a(i); 
    f(i)=f(i)/a(i);
    a(i)=1;
end

i=N;
    a(i)=a(i)-b(i-1)*c(i-1);
    f(i)=f(i)-f(i-1)*c(i-1);    
%   b(i)=b(i)/a(i); 
    f(i)=f(i)/a(i);
    a(i)=1;


for i=(N-1):-1:1
   
    f(i)=f(i)-b(i)*f(i+1);

end

% plot(real(f));hold on;
% plot(imag(f),'-r');hold off;
res=f;
% plot(m*f')