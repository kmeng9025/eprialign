function [origin, direction, chisq] = fit_line_through_points(pointlist)

[a1,a2,a3,center, svals] = principal_axis(pointlist);
origin = center;
direction = a1';
chisq = sum(svals(2:3));