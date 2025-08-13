function [ v ] = voronoi_vertices( V, Tes )
% VORONOI_VERTICES computes the Voronoi vertices.
%
%    We are given information defining the Delaunay triangulation of
%    a set of N points on the unit sphere.  
%
%    Each Delaunay triangle determines a Voronoi vertex, which is
%    the unit normal vector to that triangle.
%
%  Parameters:
%
%    Input, integer N, the number of points.
%
%    Input, real XYZ(3,N), the coordinates of the points.
%
%    Input, integer FACE_NUM, the number of Delaunay triangles.
%
%    Input, integer FACE(3,FACE_NUM), the indices of points that
%    form each Delaunay triangle.
%
%    Output, real V(3,N), the coordinates of the Voronoi vertices.
%  

  npoints = size(V,1);
  nfaces = size(Tes,1);
  
  for f = 1:nfaces

    i1 = Tes(f,1);
    i2 = Tes(f,2);
    i3 = Tes(f,3);

    v(1,f) = ( V(2,i2) - V(2,i1) ) * ( V(3,i3) - V(3,i1) ) ...
           - ( V(3,i2) - V(3,i1) ) * ( V(2,i3) - V(2,i1) );

    v(2,f) = ( xyz(3,i2) - xyz(3,i1) ) * ( xyz(1,i3) - xyz(1,i1) ) ...
           - ( xyz(1,i2) - xyz(1,i1) ) * ( xyz(3,i3) - xyz(3,i1) );

    v(3,f) = ( xyz(1,i2) - xyz(1,i1) ) * ( xyz(2,i3) - xyz(2,i1) ) ...
           - ( xyz(2,i2) - xyz(2,i1) ) * ( xyz(1,i3) - xyz(1,i1) );

    norm = sqrt ( sum ( v(1:3,f).^2 ) );

    v(1:3,f) = v(1:3,f) / norm;

  end

  return
end

