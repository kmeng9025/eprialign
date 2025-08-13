% IRADON_VOR_AREA_3D   calculate voronoi area on surface of sphere
% [areas] = iradon_vor_area_3d(kx,ky,kz);
% kx,ky,kz - k-space direction vector coordinates [double array]
% areas    - surface area on unit sphere occupied by the points
% Algorithm: Find (dtheta)*(dphi) by converting cartesian points (Gx,Gy,Gz)
%   to spherical coordinates, treating points on 2D grid of theta = 0 to pi/2
%   and phi = 0 to 2pi, and calculating voronoi area on grid. Multiply areas
%   by sin(theta) to get 'areas' (surface areas on sphere)

% Author: Gage Redler
% Center for EPR imaging in vivo physiology
% University of Chicago,OCTOBER 2013
% Contact: epri.uchicago.edu

function [areas] = iradon_vor_area_3d(un)

% convert from cartesian to spherical
[phi1, theta] = cart2sph(un(:,1),un(:,2),un(:,3));
% redefine angles for EPR notation
theta = pi/2 - theta;
phi = phi1+3*pi/2;
phi(phi>2*pi) = phi(phi>2*pi) - 2*pi;

% mirror points to completely surround original points
% right(R), left(L), up(U), down(D):
Rtheta = theta;
Rphi = (2*pi - phi) + 2*pi;
Ltheta = theta;
Lphi = -phi;
Utheta = (pi/2 - theta) + pi/2;
Uphi = phi;
Dtheta = -theta;
Dphi = phi;
RDtheta = Dtheta;
RDphi = Rphi;
RUtheta = Utheta;
RUphi = Rphi;
LDtheta = Dtheta;
LDphi = Lphi;
LUtheta = Utheta;
LUphi = Lphi;
mirroredtheta = cat(1,theta,Rtheta,Ltheta,Utheta,Dtheta,RDtheta,RUtheta,LDtheta,LUtheta);
mirroredphi = cat(1,phi,Rphi,Lphi,Uphi,Dphi,RDphi,RUphi,LDphi,LUphi);
points = cat(2,mirroredphi,mirroredtheta);

% find vertices for polygons in voronoi tesselation of points
[V,C] = voronoin(points);

% Calculate areas (A) of polygons in voronoi tesselation surrounding points
A = zeros(length(phi), 1);
for ii = 1:length(phi)
   vertX = V(C{ii},1);
   vertY = V(C{ii},2);
   A(ii) = polyarea(vertX,vertY);
end

% Multiply by sin(theta) to get surface area on unit sphere:
areas = A.*sin(theta);

%%%%%%%%%%%% SHOW PLOT WITH POINTS AND VORONOI TESSELATION: %%%%%%%%%%%%%%%
%     figure
%     plot(phi,theta,'ro')
%     hold on
%     for ii = 1:length(phi)
%        vertX = V(C{ii},1);
%        vertY = V(C{ii},2);
%        vertX = cat(1,vertX,vertX(1));
%        vertY = cat(1,vertY,vertY(1));
%        plot(vertX,vertY)
%     end
%     hold off
%     xlim([-pi/8 2*pi+pi/8])
%     ylim([-pi/8 pi/2+pi/8])
%     xlabel('Phi')
%     ylabel('Theta')
%     title('Voronoi Diagram')
%     legend('ProjPoints','VoronoiCells')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    