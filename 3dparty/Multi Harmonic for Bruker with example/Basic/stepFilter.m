function res=stepFilter(R,cutOff)
%% res=myButter(R,cutOff)
s=size(R);
L=max([s(1) s(2)]);
F=zeros(1,L);
Nc=round(cutOff*L/2);
Lc=round(L/2);
F(Lc-Nc:Lc+Nc)=1;

F=F';
for k=1:2
A=circshift(F,-2);
B=circshift(F,-1);
C=circshift(F,1);
D=circshift(F,+2);
F=A+4*B+6*F+4*C+D;
end
F=fftshift(F')/16^2;
%plot(F(1:100),'-x');

%F=1;
%%
res=zeros(s);

for i=1:s(2);
    p=R(:,i)';
    P=fft(p);
    p=ifft(P.*F);    
    res(:,i)=p';
end

%% FILTER 


