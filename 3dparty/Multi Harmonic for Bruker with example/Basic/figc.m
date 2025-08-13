function res=figc(c)
%function res=fig2(a,b)
subplot(2,1,1);
a=real(c);
imagesc(a);
subplot(2,1,2);
b=imag(c);
imagesc(b);

