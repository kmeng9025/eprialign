function z=LorFitx(Coeff,vx,vy)
k=1:length(vx);
% z=vy-1/pi*Coeff(1)* Coeff(2)./(Coeff(2)^2+vx.^2)-Coeff(3);
vx=vx-Coeff(3);
z=vy-1/pi*Coeff(1)* Coeff(2)./(Coeff(2)^2+vx.^2);