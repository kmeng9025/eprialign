function res=pltc(x)
a=real(x);
b=imag(x);
plot(a,'LineWidth',2); hold on;
plot(b,'-r','LineWidth',2); hold off;
axis tight;
res=0;