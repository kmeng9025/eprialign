function imfig = imrt_show_bev(Bev_masks, mark_bev, center, PlugSize)

if ~exist('PlugSize','var')
  PlugSize = 16;
end
if ~exist('center','var')
  center = [0,0];
end

if PlugSize == 16, Apperture = 14.2;
elseif PlugSize == 18, Apperture = 15.3;
elseif PlugSize == 20, Apperture = 18.4;
elseif PlugSize == 22, Apperture = 20.99;
else, Apperture = PlugSize;
end

imfig = figure;
fpos = get(imfig,'Position');
set(imfig,'Position',fpos+[0,0,500,0]);
pos = epr_CalcAxesPos(1,length(Bev_masks));
for ii=1:length(Bev_masks)
  axes('Position', pos(ii,:));
  show_map = zeros(size(Bev_masks{ii}.Hypoxia));
  if isfield(Bev_masks{ii}, 'Tumor')
    show_map(Bev_masks{ii}.Tumor > 0.5) = 1;
  end
  show_map(Bev_masks{ii}.Hypoxia > 0.5) = 2;
  if isfield(Bev_masks{ii}, 'Antiboost')
    show_map(Bev_masks{ii}.Antiboost > 0.5) = 3;
  elseif isfield(Bev_masks{ii}, 'Boost')
    show_map(Bev_masks{ii}.Boost > 0.5) = 3;
    show_map(Bev_masks{ii}.Hypoxia > 0.5) = show_map(Bev_masks{ii}.Hypoxia > 0.5) + 0.2;
  end
  pix_size = 0.025;
  x = (1:size(show_map,1))*pix_size; x = x - mean(x);
  imagesc(x,x,show_map);
  axis image
  axis([-10,10,-10,10])
  viscircles([0,0], Apperture/2);
  h = title(sprintf('Angle: %i  Volume: %4.2f',Bev_masks{ii}.Angle, Bev_masks{ii}.Boost_bev_volume));
  if ii == mark_bev, set(h, 'Color', 'red'); end
  
  text(0.06, 0.08, sprintf('D: %i mm (%4.1fmm)',PlugSize, Apperture), 'units','normalized','color','w');
  if isfield(Bev_masks{ii}, 'Boost')
    text(0.06, 0.92, sprintf('boost: %i', numel(find(Bev_masks{ii}.Boost))), 'units','normalized','color','w');
  end
  if isfield(Bev_masks{ii}, 'Antiboost')
    Sab = numel(find(Bev_masks{ii}.Antiboost));
    text(0.06, 0.88, sprintf('a-boost: %i (%i%%)', ...
      Sab, ...
      floor(100*numel(find(Bev_masks{ii}.Antiboost&Bev_masks{ii}.Tumor))/Sab)), ...
      'units','normalized','color','w');
  end
end
colormap jet