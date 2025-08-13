function [ xcen,ycen,r ] = circle_from_points1( points )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

myfunc=@(x)circfit1(x,points(:,1),points(:,2));
% initialize center and radius to centroid of points, and mean distance of
% points from that centroid.  this is usually close enough to get a good
% start.  actually these can be way off and still end up with a good fit.
x0=mean(points(:,1));
y0=mean(points(:,2));
xdev=points(:,1)-x0;
ydev=points(:,2)-y0;
r0=mean(sqrt(xdev.*xdev+ydev.*ydev));
mypar=[x0 y0  r0];
[xpar,chisq]=fminsearch(myfunc,mypar);
xcen=xpar(1);
ycen=xpar(2);
r=xpar(3);
end

function [sumsq ] = circfit1( par,x,y )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

xcen=par(1);
ycen=par(2);
rad=par(3);
xdev=x-xcen;
ydev=y-ycen;
rsqdev=xdev.*xdev+ydev.*ydev;
dev=sqrt(rsqdev)-rad;
sumsq=sum(dev.*dev);
end