function z=Lor(Coeff,vx)
k=1:length(vx);
z=1/pi*Coeff(1)* Coeff(2)./(Coeff(2)^2+vx.^2);
% z=1/pi*Coeff(1)* Coeff(2)./(Coeff(2)^2+vx.^2)+Coeff(3);