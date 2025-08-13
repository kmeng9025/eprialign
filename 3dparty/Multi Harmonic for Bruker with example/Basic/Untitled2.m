 % Polar plot  
 a=pi/n*1
 t=a:.001:1*pi-a;              
 
 n=10;
 polar(t,abs(sin(n*t).^10.*cos(n*t).^10));