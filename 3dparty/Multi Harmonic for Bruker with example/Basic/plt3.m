function res=plt3(a,b,c);
% blue , read, green
plot(a,'linewidth',3); hold on;
plot(b,'-r','linewidth',3);
plot(c,'-k','linewidth',3); hold off;
axis tight;
res=0;