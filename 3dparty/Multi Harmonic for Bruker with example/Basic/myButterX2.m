function res=myButterX2(R,cutOff)
% res=myButter(R,cutOff)
s=size(R);

L=max([s(1) s(2)]);

[b,a] = butter(4,cutOff,'low');
[y,x]=freqz(b,a,L/2,L/2);
filtr=abs([y'  fliplr(y')]);
F=filtr.*filtr;

res=zeros(s);

for i=1:s(2);
    p=R(:,i)';
    p=ifft(fft(p).*F);    
    res(:,i)=p';
end

%% FILTER 


