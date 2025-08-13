function [m3D] = backproject3D(Phi,Theta,Wt,x,y,z,p,ctrPt,m3D)
% Backproject - vectorized in (x,y,z,s), looping over projection
tp = x.*cos(Phi).*sin(Theta) + y.*sin(Phi).*sin(Theta) + z.*cos(Theta);
tp = tp(:) + ctrPt + 1;
a = floor(tp);
delta = tp - a;
m3D(:) = m3D(:) + Wt*(delta(:).*p(a(:)+1) + (1-delta(:)).*p(a(:)));