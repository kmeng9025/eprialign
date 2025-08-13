function res=myButterO(R,cutOff,order)
% res=myButter(R,cutOff)
s=size(R);

L=s(1);

[b,a] = butter(order,cutOff,'low');
[y,x]=freqz(b,a,L/2,L/2);
filtr=abs([y'  fliplr(y')]);


res=zeros(s);

for i=1:s(2);
    p=R(:,i)';
    p=ifft(fft(p).*filtr);    
    res(:,i)=p';
end

%% FILTER 


