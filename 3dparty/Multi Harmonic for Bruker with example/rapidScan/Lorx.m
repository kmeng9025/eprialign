function z=Lorx(Coeff,vx)
k=1:length(vx);
vx=vx-Coeff(3);
z=1/pi*Coeff(1)* Coeff(2)./(Coeff(2)^2+vx.^2);
% z=1/pi*Coeff(1)* Coeff(2)./(Coeff(2)^2+vx.^2)+Coeff(3);