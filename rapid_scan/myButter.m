function res=myButter(R,cutOff)
% res=myButter(R,cutOff)
s=size(R);

L=max([s(1) s(2)]);

[b,a] = butter(4,cutOff,'low');
[y,x]=freqz(b,a,L/2,L/2);
filtr=abs([y'  fliplr(y')]);


res=zeros(s);

for i=1:s(2);
    p=R(:,i)';
    p=ifft(fft(p).*filtr);    
    res(:,i)=p';
end

%% FILTER 


