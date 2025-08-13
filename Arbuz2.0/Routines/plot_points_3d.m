function plot_points_3d(endpoints,varargin)

style = 'r-';

if (nargin > 1), style = varargin{1}; end

x=squeeze(endpoints(:,1));
y=squeeze(endpoints(:,2));
z=squeeze(endpoints(:,3));

plot3(x,y,z,style);