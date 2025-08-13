function res=pltc2(x,y);
a=real(x);
b=imag(x);
plot(a,'LineWidth',2); hold on;
plot(b,'-r','LineWidth',2); 

a=real(y);
b=imag(y);
plot(a,'-k','LineWidth',2); 
plot(b,'-g','LineWidth',2); hold off;

axis tight;
res=0;