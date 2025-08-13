% --------------------------------------------------------------------
function arbuz_Draw2D(handles, isRaster)

% find which group we are showing
val = get(handles.pmShowGroup, 'Value');

if val == 1
  draw_list = arbuz_FindImage(handles.MainFigure, 'v', '', '', {'Ashow', 'data', 'Box', 'Color'});
elseif val > 2
  if val-2 <= length(handles.Groups)
    draw_list = arbuz_FindImage(handles, handles.Groups{val-2}.list, '', '', {'Ashow', 'data', 'Box', 'Color'});
  end
else draw_list = {};
end

if isempty(draw_list)
  disp('ArbuzGUI. Nothing to draw.'); return
end

% find transformation that leads to the frame of interest
val = get(handles.pmReferenceFrame, 'Value');
str = get(handles.pmReferenceFrame, 'String');

if val == 1
  A2frame = eye(4);
else
  find_origin = arbuz_FindImage(handles, 'all', 'FullName', str{val}, {'Ashow'});
  if isempty(find_origin), return; end
  A2frame = inv(find_origin{1}.Ashow);
end

h = figure(str2double(get(handles.eViewerFigure, 'String')));
set(h, 'Name', '2D viewer');

cla; hold on; grid on
master_draw_list = []; proxy_draw_list = [];
for ii=1:length(draw_list)
  if draw_list{ii}.ProxyIdx >= 0, proxy_draw_list{end+1}=draw_list{ii};
  else master_draw_list{end+1}=draw_list{ii};
  end
end

% draw master images first
for ii=1:length(master_draw_list)
  AA = master_draw_list{ii}.Ashow*A2frame;

  im_box = master_draw_list{ii}.Box;
  switch master_draw_list{ii}.ImageType
    case '2D',
      if isRaster
        if abs(sum(sum(eye(4)-AA))) < 1e-5
          image(master_draw_list{ii}.data)
          %         im1 = rgb2gray(im1); colormap gray;
          %         imagesc(im1, [1 max(im1(:))/6])
        else
          [im1, x, y] = project_image2D(master_draw_list{ii}.data, AA);
          image(x,y,im1)
        end
      else
        im = struct('Selected', master_draw_list{ii}.Selected);
        im.data(:,1) = [0;1;im_box(2);im_box(2);1;1;  0;1;im_box(2);  0;1;im_box(2)];
        im.data(:,2) = [5;1;1;im_box(1);im_box(1);1;  2;1;im_box(1);  2;im_box(1);1];
        im.data(:,3) = zeros(12,1);
        arbuz_DrawContour2D(gca, im, AA);
      end
  end
end

% sort the list
val = [];
for ii=1:length(proxy_draw_list)
  im_type = proxy_draw_list{ii}.ImageType;
  if strcmp(im_type, '2D')
    val(end+1) = 1;
  else
    val(end+1) = 2;
  end
end
[val,idx] = sort(val);

if isempty(idx), proxy_draw_list = {};
else proxy_draw_list = {proxy_draw_list{idx}};
end

for ii=1:length(proxy_draw_list)
  AA = proxy_draw_list{ii}.Ashow*A2frame;

  im_box = proxy_draw_list{ii}.Box;
  switch proxy_draw_list{ii}.ImageType
    case '2D',
      if isRaster
        if abs(sum(sum(eye(4)-AA))) < 1e-5
          image(proxy_draw_list{ii}.data)
        else
          [im1, x, y] = project_image2D(proxy_draw_list{ii}.data, AA);
          image(x,y,im1)
        end
      else
        im = struct('Selected', proxy_draw_list{ii}.Selected);
        im.data(:,1) = [0;1;im_box(2);im_box(2);1;1;  0;1;im_box(2);  0;1;im_box(2)];
        im.data(:,2) = [5;1;1;im_box(1);im_box(1);1;  2;1;im_box(1);  2;im_box(1);1];
        im.data(:,3) = zeros(12,1);
        arbuz_DrawContour2D(gca, im, AA);
      end
    case 'XYZ'
      arbuz_DrawPoint2D(gca, proxy_draw_list{ii}, AA);
    case 'CONTOUR'
      arbuz_DrawContour2D(gca, proxy_draw_list{ii}, AA);
  end
end

xlabel('x,mm');  ylabel('y,mm'); zlabel('z,mm');
axis image
set(gca(h), 'YDir', 'reverse')