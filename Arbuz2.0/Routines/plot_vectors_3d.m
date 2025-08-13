function plot_vectors_3d(endpoints,varargin)

color = 'r';
linestyle = '-';
if (nargin > 1), color = varargin{1}; end
if (nargin > 2), linestyle = varargin{2}; end

nvecs=size(endpoints,1);
x=reshape(endpoints(:,1),2,nvecs/2);
y=reshape(endpoints(:,2),2,nvecs/2);
z=reshape(endpoints(:,3),2,nvecs/2);

line(x,y,z, 'color',color,'linestyle', linestyle);