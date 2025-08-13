clc;
ss=size(newP);
tnp=sum(newP);
plot(sum(newP));
 
ab=-max(tnp)/min(tnp);
abS=num2str(ab);
set(t3x,'String',abS);


[mx,i]=max(tnp); [mn,j]=min(tnp);
ab=-mx/mn;
dHx=round((i-j)/ss(1)*dH)
abS=num2str(ab); dHs=num2str(dHx);
str= [dHs ,'G, A/B= ' ,abS];
set(t3x,'String',str);
