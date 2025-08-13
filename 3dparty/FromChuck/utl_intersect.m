function [ninter,alphas,intersections] = utl_intersect(contour, p1, p2)

% finds intersections of a line with a polygon.  returns parametric
% distance from first point on the line to each intersection in the vector
% "alphas", and the actual x-y coordinates of the intersections in the
% vector "intersections"
%
% C. Pelizzari Nov 2008

% first, find the projections of every contour point on a vector normal to
% the line.

theline = p2-p1;
linelength = sqrt(dot(theline,theline));
theline = theline / linelength;
thenorm = [theline(2) -1.0*theline(1)];

npoints = size(contour,1);
crossproj = zeros(npoints,1);
along = zeros(npoints, 1);
alphas = [];
intersections = [];
for i = 1:npoints
    thisvec = contour(i,:) - p1;
    crossproj(i) = dot(thisvec,thenorm);
    along(i) = dot(thisvec, theline);
end

ninter = 0;
for i = 1:npoints-1
    if(crossproj(i)*crossproj(i+1) < 0) % true if intersection between them
        %disp('negative')
        ninter = ninter+1;
        % so at this point, crossproj(i) and (i+1) differ in sign.  It goes
        % to zero somewhere in between, which is our intersection point
        % (where the projection of a point on the contour edge lies right
        % on our line).  just need to move the right distance along the
        % edge.
        % regardless whether crossproj(i) is positive or negative, we want
        % to know the fraction of the distance from i to i+1 it accounts
        % for.  Can get this by taking absolute value of the ratio.  Note
        % that if we didn't do this, we could end up with a negative
        % fraction which would be incorrect.
        frac = abs(crossproj(i) / (crossproj(i+1) -crossproj(i)));
        dist = along(i) + frac*(along(i+1) - along(i));
        alphas(ninter) = dist / linelength;
        intersections(ninter,:) = p1+alphas(ninter)*(p2-p1);
    end
    if (crossproj(i+1) == 0) % line goes through the vertex
        %disp('zero')
        ninter = ninter+1;
        dist = along(i+1);
        alphas(ninter) = dist / linelength;
        intersections(ninter,:) = contour(i,:);
    end
end
    
