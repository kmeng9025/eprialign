function chisq = point_line_residual(pointlist)

nlines = size(pointlist, 3);
chisq=0;
for l = 1:nlines
    mypoints=pointlist(:,:,l);
    [o,d,x]=fit_line_through_points(mypoints);
    chisq=chisq+x;
end