function mse=MSE(Pideal,P)

 x0=(Pideal-P).^2;
 x=sum(x0);
 x1=sum(x');
 
 y0=Pideal.^2;
 y=sum(y0);
 y1=sum(y');
 
 mse=x1/y1;
 
 