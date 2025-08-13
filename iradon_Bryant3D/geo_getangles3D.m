function [Rec] = geo_getangles3D(Pars,Rec)
%GEO_GETANGLES 

x = Pars.GradX;
y = Pars.GradY;
z = Pars.GradZ;
r = sqrt(x.^2+y.^2+z.^2);
Rec.Theta = acos(z./r);
Rec.Phi = atan2(y,x);

%Rec.Phi = acot(z./sqrt(x.^2+y.^2));
%Rec.Theta = 2*acot((sqrt(x.^2+y.^2)+x)./y);
%[Rec.Phi,Rec.Theta,R] = cart2sph(x,y,z);


