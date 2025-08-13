function [] = arbuz_HistoSliceRDR(hGUI, options)

figure(options.FigN); 
set(options.FigN,'Name','Preparing the content ...'); drawnow;

% get number of objects to render
nSlices = length(options.slices);
nImages = length(options.images);

plist = {'Ashow','data','FullName','Color','Bbox','Mask','SlaveList', 'Anative'};
find_list = cell(nImages, 1);
for ii=1:nImages
  res = arbuz_FindImage(hGUI, 'master', 'Name', options.images{ii}.name, plist);
  find_list{ii} = res{1};
end
slice_list = cell(nSlices, 1);
for ii=1:nSlices
  res = arbuz_FindImage(hGUI, 'master', 'Name', options.slices{ii}.name, plist);
  slice_list{ii} = res{1};
end

% This is the bounding box
boundsX = options.bounds1;
boundsY = options.bounds2;
maxx = max(boundsX); minx = min(boundsX);
maxy = max(boundsY); miny = min(boundsY);

for ii=1:nSlices
  Aslice2Show = slice_list{ii}.Ashow;

  % 2D image
%   corners = [minx,miny,0; minx,maxy,0; maxx,maxy,0; maxx,miny,0];
%   tcorners= fix(htransform_vectors(inv(Aslice2Show),corners));
%   tcorners(:,3) = 1;
  
  tcorners = [minx,miny,0; minx,maxy,0; maxx,maxy,0; maxx,miny,0];
  
  [slice_list{ii}.im, xp, yp]=resample_plane(slice_list{ii}.data, tcorners, length(boundsY), length(boundsX));
  zp = zeros(size(xp));
  
  % 3D image
  for jj=1:nImages
    AA = Aslice2Show*inv(find_list{jj}.Ashow);
    
    xyzvol = [xp(:) yp(:) zp(:)];
    xyzvt = htransform_vectors(AA, xyzvol);
    xt = reshape(xyzvt(:,1), size(xp));
    yt = reshape(xyzvt(:,2), size(yp));
    zt = reshape(xyzvt(:,3), size(zp));
    
    find_list{jj}.im{ii}  = interp3(cast(find_list{jj}.data,'double'), xt,yt,zt, 'linear',0);
    %   %
    %   PostProcessing = safeget(find_list{jj}.Color, 'PostProcessing','');
    %   if ~isempty(PostProcessing)
    %     img = find_list{jj}.im; %#ok<NASGU>
    %     find_list{jj}.im = eval(PostProcessing);
    %   end
    %   %
    the_mask = safeget(options.images{jj}, 'mask', 'native');
    the_scale = safeget(options.images{jj}, 'scale', [0,1]);
    switch the_mask
      case {''}, find_list{jj}.im_mask{ii} = [];
      case {'native'}
        if ~isempty(find_list{jj}.Mask)
          find_list{jj}.im_mask{ii}  = interp3(cast(find_list{jj}.Mask,'double'), xt,yt,zt, 'linear',0) > 0.5;
%                   switch handles.options.SliceDir
%                     case 'X', find_list{jj}.im_mask = permute(find_list{jj}.im_mask, [1,3,2]);
%                     case 'Y', find_list{jj}.im_mask = permute(find_list{jj}.im_mask, [2,3,1]);
%                   end
%                   SliceErode=safeget(find_list{jj}.Color, 'SliceErode', 0);
%                   if SliceErode > 0
%                     se = strel('disk', SliceErode);
%                     find_list{jj}.im_mask = imerode(find_list{jj}.im_mask,se);
%                   end
%                   if safeget(find_list{jj}.Color, 'SliceLargest', 0)
%                     for pp=1:nSlices
%                       [labeled_mask, nMask] = bwlabel(find_list{jj}.im_mask(:,:,pp));
%                       mask_square = zeros(nMask, 1);
%                       for ii=1:nMask
%                         mask_square(ii) = numel(find(labeled_mask == ii));
%                       end
%                       [mm, max_idx] = max(mask_square);
%                       find_list{jj}.im_mask(:,:,pp) = labeled_mask == max_idx;
%                     end
%                   end
        else
          find_list{jj}.im_mask{ii} = [];
        end
      case 'minmax'
        find_list{jj}.im_mask{ii} = find_list{jj}.im{ii} >= min(the_scale) & find_list{jj}.im{ii} <= max(the_scale);
      case 'minmax_alpha'
        the_mask = find_list{jj}.im{ii} >= min(the_scale) & find_list{jj}.im{ii} <= max(the_scale);
        
        %       find_list{jj}.im_mask = (find_list{jj}.im - min(the_scale))/diff(the_scale);
        
        find_list{jj}.im_mask{ii} = sin((find_list{jj}.im{ii} - min(the_scale))/diff(the_scale)*pi/2);
        find_list{jj}.im_mask{ii}(~the_mask) = 0;
      otherwise
        %       % get the mask
        %       find_mask = arbuz_FindImage(handles.hh, 'all', 'Name', the_mask, {'data'});
        %       pidx = -1;
        %       for ii=1:length(find_mask)
        %         if strcmp(find_list{jj}.Image, find_mask{ii}.Image), pidx = ii; break; end
        %       end
        %       if pidx > 0 && isequal(size(find_mask{pidx}.data), size(find_list{jj}.data))
        %         if ~isempty(find_list{jj}.Mask)
        %           find_mask{pidx}.data = find_mask{pidx}.data & find_list{jj}.Mask;
        %         end
        %         find_list{jj}.im_mask  = interp3(cast(find_mask{pidx}.data,'double'), xt,yt,zt, 'linear',0) > 0.5;
        %         switch handles.options.SliceDir
        %           case 'X', find_list{jj}.im_mask = permute(find_list{jj}.im_mask, [1,3,2]);
        %           case 'Y', find_list{jj}.im_mask = permute(find_list{jj}.im_mask, [2,3,1]);
        %         end
        %       else
        %         find_list{jj}.im_mask = [];
        %       end
        
        HighCutOff = safeget(find_list{jj}.Color, 'HighCutOff', 0.95);
        LowCutOff = safeget(find_list{jj}.Color, 'LowCutOff', 0.05);
        CutOffAbsoluteScale = safeget(find_list{jj}.Color, 'CutOffAbsoluteScale', 0);
        
        mx =  cast(max(find_list{jj}.im{1}(:)), 'double');
        
        find_list{jj}.scale = [LowCutOff,HighCutOff];
        find_list{jj}.clevel = safeget(find_list{jj}.Color, 'ContourThreshold', 0.5);
        
        if ~CutOffAbsoluteScale,
          find_list{jj}.scale = find_list{jj}.scale * mx;
          find_list{jj}.clevel = find_list{jj}.clevel * mx;
        end
    end
    
    % generate contours
    the_contour = safeget(options.images{jj}, 'contour', '');
    how_to_mask = safeget(options.images{jj}, 'how_to_mask', 1);
    contour_level = safeget(options.images{jj}, 'contour_level', [0.5,0,0]);
    contour_index = safeget(options.images{jj}, 'contour_index', [true,false,false]);
    contour_level = contour_level(contour_index);
    
    image_for_cc = [];
    find_list{jj}.cbar_contours = {};
    if ~isempty(the_contour)
      idxContour = find(contour_index);
      nLevels = numel(idxContour);
      if nLevels == 0, break; end
      switch the_contour
        case 'self'
          image_for_cc = find_list{jj}.im{ii};
        otherwise
          % get the mask
          find_mask = arbuz_FindImage(hGUI, find_list{jj}.SlaveList, 'Name', the_contour, {'data'});
          if ~isempty(find_mask) && isequal(size(find_mask{1}.data), size(find_list{jj}.data))
            if ~isempty(find_list{jj}.Mask)
              find_mask{1}.data = find_mask{1}.data & find_list{jj}.Mask;
            end
            image_for_cc  = interp3(cast(find_mask{1}.data,'double'), xt,yt,zt, 'linear',0) > 0.5;
%             switch handles.options.SliceDir
%               case 'X', image_for_cc = permute(image_for_cc, [1,3,2]);
%               case 'Y', image_for_cc = permute(image_for_cc, [2,3,1]);
%             end
            contour_level = 0.5;
          end
      end
    end
    
    if ~isempty(image_for_cc)
      find_list{jj}.clevel = contour_level;
      find_list{jj}.c_width = ...
        safeget(options.images{jj}, 'contour_linewidth', 0.5);
      find_list{jj}.c_color = ...
        safeget(options.images{jj},'contour_colors', [1,1,1;1,1,1;1,1,1]);
      find_list{jj}.c_color = find_list{jj}.c_color(contour_index,:);
      
      % colorbar settings
      find_list{jj}.cbar_contours = cell(1, length(find_list{jj}.clevel));
      for kk=1:length(find_list{jj}.clevel)
        find_list{jj}.cbar_contours{kk}.level = find_list{jj}.clevel(kk);
        find_list{jj}.cbar_contours{kk}.color = find_list{jj}.c_color(kk,:);
        find_list{jj}.cbar_contours{kk}.linewidth = find_list{jj}.c_width;
      end
      
      image_for_cc = double(image_for_cc);
%       if ~isempty(find_list{jj}.im_mask{ii})
%         switch how_to_mask
%           case 1, image_for_cc(~(image_for_cc > 0.5)) = 1E6;
%           case 2, image_for_cc(~(image_for_cc > 0.5)) = -1E6;
%         end
%       end
      
      if any(any(image_for_cc))
        if numel(contour_level) > 1
          find_list{jj}.cc{ii} = contourc(boundsY, boundsX, image_for_cc,...
            contour_level);
        else
          find_list{jj}.cc{ii} = contourc(boundsY, boundsX, image_for_cc,...
            contour_level*[1,1]);
        end
      else
        find_list{jj}.cc{ii} = [];
      end
    end
  end
end


% create figure layout
print_style = 2;
if nImages < 1, print_style = 1; end
figure(options.FigN); clf;
set(options.FigN,'Name',sprintf('%s -- ', arbuz_get(hGUI, 'FILENAME')));
nrows = iff(print_style == 2, 2, 1);
ax_coordinates = epr_CalcAxesPos(nrows, nSlices, [0.0025 0.005], [0.01 0.04]);
h = zeros(nrows, nSlices);

% draw all
for ii = 1:nSlices
  % create axis
  h(1, ii) = axes('Position', ax_coordinates(ii,:));
  if print_style == 2
    h(2, ii) = axes('Position', ax_coordinates(ii+nSlices,:));
    hold(h(2, ii), 'on')
  end
  title(h(1, ii), sprintf('%s', options.slices{ii}.name));
  hold(h(1, ii), 'on')

  % 2D image
  if isempty(options.slices{ii}.scale)
    imagesc( boundsY, boundsX, slice_list{ii}.im, 'Parent', h(1, ii));
  else
    imagesc( boundsY, boundsX, slice_list{ii}.im, 'Parent', h(1, ii));
    set(h(1, ii),'CLim',options.slices{ii}.scale);
  end
  
  for jj=1:nImages
    the_colormap = safeget(options.images{jj}, 'colormap', 'jet');
    the_scale = safeget(options.images{jj}, 'scale', [0,1]);
    
    if print_style == 2
     image3D_panel = h(2, ii);
    else
     image3D_panel = h(1, ii);
    end
    
    % 3D image
    rgb = arbuz_ind2rgb(find_list{jj}.im{ii}, the_colormap, the_scale);
    hhh = image(boundsY, boundsX, rgb, 'Parent', image3D_panel, 'CDataMapping','scaled');
    alpha = safeget(options.images{jj}, 'alpha', 0.5);
    if ~isempty(find_list{jj}.im_mask{ii})
      set(hhh,'AlphaData',alpha*find_list{jj}.im_mask{ii});
    else
      alpha_mask = alpha*ones(size(find_list{jj}.im{ii}));
      set(hhh,'AlphaData',alpha_mask);
    end
    %
    %     % colorbar
    %     the_colorbar = safeget(handles.options.images{jj}, 'colorbar', 1);
    %     cbar_ax = get(current_ax,'Position');
    %     switch the_colorbar
    %       case 2
    %         cbar_ax = cbar_ax + [4.8*cbar_ax(3)/6, 1*cbar_ax(4)/6, -5*cbar_ax(3)/6, -2*cbar_ax(4)/6];
    %         colorbar_ax = axes('Position', cbar_ax);
    %         ShowColorbarFIG(colorbar_ax, 'fixed',  handles.options.images{jj}.scale, 'text', 'center',...
    %           'colormap', the_colormap, 'contours', find_list{jj}.cbar_contours)
    %       case 3
    %         cbar_ax = cbar_ax + [0.2*cbar_ax(3)/6, 1*cbar_ax(4)/6, -5*cbar_ax(3)/6, -2*cbar_ax(4)/6];
    %         colorbar_ax = axes('Position', cbar_ax);
    %         ShowColorbarFIG(colorbar_ax, 'fixed', handles.options.images{jj}.scale, 'text', 'center', ...
    %           'colormap', the_colormap, 'contours', find_list{jj}.cbar_contours)
    %     end
  end
end

% draw contours
for ii = 1:nSlices
  for jj = 1:nImages
    if isfield(find_list{jj}, 'cc')
      contour_data=find_list{jj}.cc{ii};
      ll=1;
      while ll < size(contour_data,2)
        p = contour_data(2,ll); lev = contour_data(1,ll);
        r1 = contour_data(:,ll+1:ll+p); ll = ll+1+p;
        lev_idx = find_list{jj}.clevel == lev;
        lev_color = find_list{jj}.c_color(lev_idx,:);
        plot(r1(1,:),r1(2,:),'Parent', h(1, ii), ...
          'Color', lev_color, 'Linewidth', find_list{jj}.c_width);
      end
    end
  end
end

set(h(:), 'Box', 'on', 'DataAspectRatio', [1,1,1])
set(h(:), 'yTick', []);
axis(h(:), 'image');
