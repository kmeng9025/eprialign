function z=LorFit(Coeff,vx,vy)
%k=1:length(vx);
% z=vy-1/pi*Coeff(1)* Coeff(2)./(Coeff(2)^2+vx.^2)-Coeff(3);
A= -1./(1i*Coeff(2)+vx);
z=vy-1/pi*Coeff(1)*A;
z=abs(z);