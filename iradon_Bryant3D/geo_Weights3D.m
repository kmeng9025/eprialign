function [p,Pars,Rec] = geo_Weights3D(p,Pars,Rec)
%GEO_GETWEIGHTS3D Summary of this function goes here
%   Detailed explanation goes here

[x,y,z] = sph2cart(Rec.Phi,pi/2-Rec.Theta,1);
dsites = [x,y,z];
npoints = size(dsites,1);
minz = min(z);
ind = find(z <= 1.5*minz);
esites = [dsites; -x(ind), -y(ind), -z(ind)];
esites = [esites; 0,0,-0.5];
dt = delaunayTriangulation(esites(:,1),esites(:,2),esites(:,3));
ch = dt.freeBoundary();
[row, col] = find(ch == size(esites,1));
ind = unique(row);
ch(ind,:) = [];
nfaces = size(ch,1);
esites(end,:) = [];
Tri = triangulation(ch,esites(:,1),esites(:,2),esites(:,3));
vsites = circumcenter(Tri);
vsites = recon_normr(vsites);
Si = vertexAttachments(Tri);

wt = zeros(npoints,1);
for i = 1:npoints
  for j = 1:length(Si{i})-2
    s = geo_TriangleArea([vsites(Si{i}(1),:);vsites(Si{i}(j+1),:);vsites(Si{i}(j+2),:)]);
    if abs(s) >  1E-14
      area = stri_vertices_to_area(1.0,vsites(Si{i}(1),:)',...
        vsites(Si{i}(j+1),:)', vsites(Si{i}(j+2),:)');
      wt(i) = wt(i) + area;
    end
  end
end
Rec.Wt = wt;
Rec.nProj = size(wt,1);