function figN = imrt_show_segmentation(figN, data, presentation, slice_range, opts)

legend   = safeget(opts, 'legend', '');
show_max = safeget(opts, 'show_max', max(data(:)));
show_min = safeget(opts, 'show_min', min(data(:)));
slicedir = safeget(opts, 'slicedir', 3);

figN = figure(figN); clf;
pos = epr_CalcAxesPos(2,3,[0.01,0.01],[0.2,0.06]);
pos2 = epr_CalcAxesPos(4,6,[0.005,0.005],[0.04,0.07]);
show_idx=fix(linspace(min(slice_range),max(slice_range),6));
show_idx_left = min(slice_range) - [4,3,2,1];
while(min(show_idx_left) < 1), show_idx_left = show_idx_left + 1; end 
show_idx_right = max(slice_range) + [1,2,3,4];
while(max(show_idx_right) > size(data, 3)), show_idx_right = show_idx_right - 1; end 

for jj=1:length(show_idx)
  axes('position',pos(jj,:));
  idx = show_idx(jj);
  
  rgb = arbuz_ind2rgb(get_slice(data, idx, slicedir), 'gray', [show_min,show_max]);
  image(rgb, 'CDataMapping','scaled'); hold on
  rgb = arbuz_ind2rgb(get_slice(presentation, idx, slicedir), 'jet', [0,5]);
  image(rgb, 'AlphaData',get_slice(presentation, idx, slicedir) > 0.5); hold on
  
  text(0.5, 0.95, sprintf('slice = %i',idx),'units','normalized',...
    'HorizontalAlignment','center','color','w','fontsize',14);
  axis image; colormap jet;
  if jj==2, title(legend); end
end
for jj=1:length(show_idx_left)
  axes('position',pos2((jj-1)*6+1,:));
  idx = show_idx_left(jj);
  rgb = arbuz_ind2rgb(get_slice(data, idx, slicedir), 'gray', [show_min,show_max]);
  image(rgb, 'CDataMapping','scaled'); hold on
  rgb = arbuz_ind2rgb(get_slice(presentation, idx, slicedir), 'jet', [0,5]);
  image(rgb, 'AlphaData',get_slice(presentation, idx, slicedir) > 0.5); hold on
  text(0.5, 0.95, sprintf('slice = %i',idx),'units','normalized',...
    'HorizontalAlignment','center','color','w','fontsize',14);
  axis image; colormap jet;
end
for jj=1:length(show_idx_right)
  axes('position',pos2((jj-1)*6+6,:));
  idx = show_idx_right(jj);
  rgb = arbuz_ind2rgb(get_slice(data, idx, slicedir), 'gray', [show_min,show_max]);
  image(rgb, 'CDataMapping','scaled'); hold on
  rgb = arbuz_ind2rgb(get_slice(presentation, idx, slicedir), 'jet', [0,5]);
  image(rgb, 'AlphaData',get_slice(presentation, idx, slicedir) > 0.5); hold on
  text(0.5, 0.95, sprintf('slice = %i',idx),'units','normalized',...
    'HorizontalAlignment','center','color','w','fontsize',14);
  axis image; colormap jet;
end

function res = get_slice(data, idx, dir)
switch dir
  case 1, res = squeeze(data(idx, :, :));
  case 2, res = squeeze(data(:, idx, :));
  case 3, res = squeeze(data(:, :, idx));
end