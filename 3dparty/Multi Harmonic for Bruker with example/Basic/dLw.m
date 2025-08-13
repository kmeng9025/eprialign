function [dx dy]=dLw(x,y)
%[dx dy]=dLw(x,y)
[ymin xmin]=min(y);
[ymax xmax]=max(y);
dx=(xmax-xmin)*(x(2)-x(1));
dy=ymax-ymin;
dx=abs(dx);