
function [p,Pars,Rec] = interp_tri3D(p,Pars,Rec)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[x,y,z] = sph2cart(Rec.Phi,pi/2-Rec.Theta,1);
dsites = [x,y,z];
npoints = size(dsites,1);
minz = min(z);
ind = find(z <= 1.5*minz);
esites = [dsites; -x(ind), -y(ind), -z(ind)];
plength = size(p,1);
pe = [p,p(plength:-1:1,ind)];
dt = DelaunayTri(esites(:,1),esites(:,2),esites(:,3));
ch = dt.freeBoundary();
[row col] = find(ch == size(esites,1));
ch(row,:) = [];
nfaces = size(ch,1);

for i = 1:nfaces
  r = esites(ch(i,:)',:);
  b = [1/3,1/3,1/3];
  s = r'*b';
  isites(i,:) = s;
  p(:,npoints+i) = mean(pe(:,ch(i,:)),2);
end

idx = find(isites(:,3) < 0);
isites(idx,:) = [];
p(:,npoints+idx) = [];
[Phi,Theta,R] = cart2sph(isites(:,1),isites(:,2),isites(:,3));
Rec.Phi = [Rec.Phi; Phi];
Rec.Theta = [Rec.Theta; pi/2-Theta];
Rec.nProj = size(dsites,1)+size(isites,1);