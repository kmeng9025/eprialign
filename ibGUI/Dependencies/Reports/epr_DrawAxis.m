% function epr_DrawAxis(hh, x, y, xy, xy_mask, xy_mask_add, guides, txt, opt)
% hh     - axis handle
% x,y,xy - 2D image xy and its axis
% xy_mask - image alpha mask, can be empty
% xy_mask_add - additional mask, can be empty
% guides  - [x,y] cross-guides to show some point in the image
% txt     - text shown in the lower bottom corner
% opt     - additional options structure
%     ext_mask  - 'transparency' or 'wire' - the way to display additional mask
%     clim      - [min,max] of image colormap
%     ShowGuide - 'no' for no guides or see plot function for linestyles
%     GuideLineWidth - linewidth of guides

% Boris Epel (c) 2007-2010
% University of Chicago
% bepel@uchicago.edu

function epr_DrawAxis(hh, x, y, xy, xy_mask, xy_mask_add, guides, txt, opt)

clim = safeget(opt, 'clim', []);
ext_mask = safeget(opt, 'ext_mask', 'wire');
if isempty(clim), clim = [min(xy(:)), max(xy(:))]; end

hold(hh, 'off'); % clear axis
h = image(x, y, xy, 'Parent', hh, 'CDataMapping','scaled');
set(hh,'CLim',clim, 'Box', 'on', 'DataAspectRatio', [1,1,1])
hold(hh, 'on');
if  mean(diff(x(:))) > 0,  set(hh, 'XDir', 'normal'); else set(hh, 'XDir', 'reverse'); end
if  mean(diff(y(:))) > 0,  set(hh, 'YDir', 'normal'); else set(hh, 'YDir', 'reverse'); end

guide_style = safeget(opt, 'ShowGuide', 'k:');
if isempty(strfind(guide_style, 'no')) && ~isempty(guides)
  t_mnx = min(x); t_mny = min(y); t_mxx = max(x); t_mxy = max(y);
  lx = t_mxx-t_mnx; ly = t_mxy-t_mny;
  xtick = lx/10;  ytick = ly/10;
  guide_LW = safeget(opt, 'GuideLineWidth', 2);
  plot(t_mnx+[0,xtick], guides(2)*[1,1],guide_style, 'LineWidth', guide_LW, 'Parent', hh)
  plot(t_mxx-[xtick,0], guides(2)*[1,1],guide_style, 'LineWidth', guide_LW, 'Parent', hh)
  plot(guides(1)*[1,1], t_mny+[0,ytick],guide_style, 'LineWidth', guide_LW, 'Parent', hh)
  plot(guides(1)*[1,1], t_mxy-[ytick,0],guide_style, 'LineWidth', guide_LW, 'Parent', hh)
end

text(.03,.04,[' ',txt],'Parent', hh, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left',...
  'Units','normalized','FontWeight','bold');

if ~iscell(xy_mask_add)
  m = struct('Mask', xy_mask_add);
  xy_mask_add = {}; xy_mask_add{1} = m;
end

ax_cmap = safeget(opt, 'colormap', []);

if ~isempty(xy_mask)
  alphadatImg = zeros(size(xy_mask));
  
  if strcmp(ext_mask, 'transparency')
    m = alphadatImg;
    for ii=1:length(xy_mask_add)
      if ~isempty(xy_mask_add{ii}.Mask), m = m & xy_mask_add{ii}.Mask; end
    end
    alphadatImg(m & xy_mask)=1;
    alphadatImg(xy_mask & (~m))=0.35;
  else
    alphadatImg(xy_mask)=1;
  end
  set(h,'AlphaData',alphadatImg);
end

if ~strcmp(ext_mask, 'transparency')
  for ii=1:length(xy_mask_add)
    % this mask is not all-no or all-yes
    if any(xy_mask_add{ii}.Mask(:)) && any(~xy_mask_add{ii}.Mask(:))
      lw = safeget(xy_mask_add{ii}, 'LineWidth', 2);
      cl = safeget(xy_mask_add{ii}, 'Color', 'k');
      contour(x,y,double(xy_mask_add{ii}.Mask), 0.5, cl, 'Linewidth',lw,'Parent', hh);
    end
  end
end

if safeget(opt, 'isShowContourLegend', 0) && ~isempty(xy_mask_add)
  pos = get(hh, 'Position');
  for ii=1:length(xy_mask_add)
    text(.05, 1 - (15/pos(4))*ii, safeget(xy_mask_add{ii}, 'Name', ''), ...
      'Color', safeget(xy_mask_add{ii}, 'Color', 'r'),...
      'interpreter', 'none', 'units', 'normalized', 'Parent', hh );
  end
end

axis(hh, 'square');
if ~isempty(ax_cmap),  colormap(hh, eval(ax_cmap)); end
