clc
subplot(2, 1,1);
plot(a); hold on; 
a=zeroLine(a',0.05)';

a=a.*hamming(151)';
a=a.*hamming(151)';

k=kern/sum(kern);
plot(k,'-r'); 

D=myConv(a',k')'; 
plot(D);hold off;

subplot(2, 1,2);

K=deco(D,a,0.1);
plot(real(K));

