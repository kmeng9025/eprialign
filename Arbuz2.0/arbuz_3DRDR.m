% arbuz_3DRDR(hGUI, options);
% handles - [structure] of handles for the GUI
% hGUI - handle to the object that holds the project [double]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function []=arbuz_3DRDR(hGUI, options)

h = figure(options.FigN); 
if ishandle(options.FigN)
  CameraPosition = get(gca(h), 'CameraPosition');
  CameraTarget = get(gca(h), 'CameraTarget');
end
set(options.FigN,'Name','Preparing the content ...'); drawnow;

if isempty(options.images)
  arbuz_ShowMessage(hGUI, 'ArbuzGUI. Nothing to draw.'); return
end

plist = {'Ashow','data','FullName','Color','Bbox','Mask','SlaveList'};
options.images = arbuz_FindImage(hGUI, options.images, '', '', plist);

ha = gca(h);
cla(ha); hold(ha, 'on'); grid(ha, 'on')

proxy_draw_list = [];
master_draw_list = [];
for ii=1:length(options.images)
  if options.images{ii}.SlaveIdx >= 0, proxy_draw_list{end+1}=options.images{ii};
  else master_draw_list{end+1}=options.images{ii};
  end
end
    
% draw master images first
for ii=1:length(master_draw_list)
  AA = master_draw_list{ii}.Ashow*options.A2frame;

  im_box = master_draw_list{ii}.Box;
  switch master_draw_list{ii}.ImageType
    case '2D'
      im = struct('Selected', master_draw_list{ii}.Selected);
      im.data(:,1) = [0;1;im_box(2);im_box(2);1;1;  0;1;im_box(2);  0;1;im_box(2)];
      im.data(:,2) = [5;1;1;im_box(1);im_box(1);1;  2;1;im_box(1);  2;im_box(1);1];
      im.data(:,3) = zeros(12,1);
      im.Color = safeget(master_draw_list{ii}, 'Color', []);
      arbuz_DrawContour3D(gca, im, AA);
    case '3DSURFACE'
      arbuz_DrawSurface3D(gca, master_draw_list{ii}, AA);
    case 'CONTOUR'
      arbuz_DrawContour3D(gca, master_draw_list{ii}, AA);
  end 
end

% proxy imager
for ii=1:length(proxy_draw_list)
  AA = proxy_draw_list{ii}.Ashow*options.A2frame;

  im_box = proxy_draw_list{ii}.Box;
  switch proxy_draw_list{ii}.ImageType
    case '2D'
      clear im
      im.data(:,1) = [0;1;im_box(2);im_box(2);1;1;  0;1;im_box(2);  0;1;im_box(2); 0;20;1];
      im.data(:,2) = [5;1;1;im_box(1);im_box(1);1;  2;1;im_box(1);  2;im_box(1);1; 2;1;20];
      im.data(:,3) = zeros(15,1);
      im.Color = safeget(proxy_draw_list{ii}, 'Color', []);
      arbuz_DrawContour3D(gca, im, AA);
    case 'XYZ'
      arbuz_DrawPoint3D(gca, proxy_draw_list{ii}, AA);
    case 'CONTOUR'
      arbuz_DrawContour3D(gca, proxy_draw_list{ii}, AA);
    case '3DSURFACE'
      arbuz_DrawSurface3D(gca, proxy_draw_list{ii}, AA);
  end
end

Coordinates = arbuz_get(hGUI, 'Coordinates');
for ii=1:length(Coordinates)
  clear cc
  cc.data(:,1) = [0;0;8;0;0;0;0;0;0];
  cc.data(:,2) = [2;0;0;2;0;8;2;0;0];
  cc.data(:,3) = [0;0;0;0;0;0;0;-8;8];
  arbuz_DrawContour3D(gca, cc, Coordinates{ii}.A);
end

xlabel(ha, 'x,mm');  ylabel(ha, 'y,mm'); zlabel(ha, 'z,mm');
axis(ha, 'equal')
axis(ha, 'tight')
grid(ha, 'on');

set(ha, 'xdir', 'normal', 'ydir', 'normal', 'zdir', 'normal');
if exist('CameraPosition', 'var')
  set(ha, 'CameraPosition',CameraPosition);
  set(ha, 'CameraTarget',CameraTarget);
end

set(h, 'Name', '3D viewer: Ready.');

cameratoolbar(h,'NoReset')
cameratoolbar(h,'Show')
cameratoolbar(h,'SetMode','orbit')
cameratoolbar(h,'SetCoordSys','y')
drawnow

FigureNavigationPanelPNL('create', options.FigN);
