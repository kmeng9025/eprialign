% function ShowColorbarFIG(fig, type, min_max, varargin)
% fig  - fig number or axis or [] for new figure
% type - 'left'/'center'/'right'/'manual'
% arguments
%   'position' - axis position

function ShowColorbarFIG(fig, type, m_m, varargin)

if nargin<3
  help ShowColorbarFIG;
  return
elseif rem(nargin-3,2)~=0
  error('Property/value must be pairs!');
end

args = [];
for k = 1:2:nargin-4
  args.(lower(varargin{k})) = varargin{k+1};
end

mmin = m_m(1);
mmax = m_m(2);
d_mm = mmax-mmin;

switch safeget(args, 'colormap', '?')
  case 'jet', the_colormap = jet;
  case 'HSV', the_colormap = HSV;
  case 'hot', the_colormap = hot;
  case 'cool', the_colormap = cool;
  case 'bone', the_colormap = bone;
  case 'autumn', the_colormap = autumn;
  case 'spring', the_colormap = spring;
  case 'summer', the_colormap = summer;
  case 'winter', the_colormap = winter;
  case 'gray', the_colormap = gray;
  case 'copper', the_colormap = copper;
  case 'pink', the_colormap = pink;
  otherwise, the_colormap = colormap(fig);
end

if ishandle(fig) && isempty(fig)
  fig = figure; h_a = [];
elseif ishandle(fig) && strcmp(get(fig, 'Type'), 'figure')
  figure(fig); h_a = [];
elseif ishandle(fig) && strcmp(get(fig, 'Type'), 'axes')
  h_a = fig;
  fig = get(h_a, 'Parent');
end

title_text = safeget(args, 'title', '');
text_pos = safeget(args, 'text', 'center');
if ~strcmp(type, 'fixed')
  h_a = axes('Position', axis_position(fig, type, args));
else
  delete(get(h_a, 'Children'));
end
hold(h_a, 'on');

cbar_contours = safeget(args, 'contours', {});
switch type
  case {'manual', 'fixed'}
    map = the_colormap; im(:,1,:) = map; im(:,2,:) = map;
    sz = size(map, 1);
    image(1:2, mmin+(0:sz-1) * (mmax-mmin)/(sz-1), uint8(fix(im*255)), 'Parent', h_a);
    plot([0.5,2.5,2.5,0.5,0.5], [mmin,mmin,mmax,mmax,mmin], 'k', 'Linewidth', 1, 'Parent',h_a);
    set(h_a, 'XTick', [], 'YTick', [], 'Box', 'off', 'YDir', 'normal', 'LineWidth', 1);
    axis(h_a, 'tight');
    switch text_pos
      case 'center'
        im_ticks = calc_ticks(mmin, mmax);
        mark = 0.2;
        for ii=1:length(im_ticks)
          plot(0.5 + [0,mark], im_ticks(ii)*[1,1],'k', 'Parent',h_a)
          plot(2.5 + [-mark, 0], im_ticks(ii)*[1,1],'k', 'Parent',h_a)
          text(1.5, im_ticks(ii), sprintf('%g',im_ticks(ii)), 'HorizontalAlignment', 'center', 'FontWeight', 'bold','Parent',h_a);
        end
        title(title_text)
      case 'right'
        set(h_a, 'TickLength', [0.04 0.04]);
        set(h_a, 'TickDir', 'in', 'YAxisLocation', 'right', 'YTickLabelMode', 'auto');
        title(title_text)
      case 'none'
        map = the_colormap; im(1,:,:) = map; im(2,:,:) = map;
        sz = size(map, 2);
        image(mmin+(0:sz-1) * (mmax-mmin)/(sz-1), 1:2, uint8(fix(im*255)), 'Parent', h_a);
        set(h_a, 'YTick', [], 'TickDir', 'in', 'Box', 'on', 'YDir', 'normal');
        set(h_a, 'XTickLabel', [], 'TickLength', [0.016 0.025]);
    end
    for kk = 1:length(cbar_contours)
      hh = plot([0.5,2.5], cbar_contours{kk}.level*[1,1], 'Parent',h_a);
      set(hh, 'Color', cbar_contours{kk}.color,'LineWidth',cbar_contours{kk}.linewidth*2)
    end
  case 'top'
    map = the_colormap; im(1,:,:) = map; im(2,:,:) = map;
    sz = size(map, 2);
    image(mmin+(0:sz-1) * (mmax-mmin)/(sz-1), 1:2, uint8(fix(im*255)));
    set(h_a, 'YTick', [], 'TickDir', 'in', 'Box', 'on', 'YDir', 'normal');
    im_ticks = get(h_a, 'XTick'); im_ticks = im_ticks(2:end-1);
    set(h_a, 'XTickLabel', [], 'TickLength', [0.016 0.025]);
    for ii=1:length(im_ticks)
      text(im_ticks(ii), 1.5, sprintf('%f',im_ticks(ii)), 'HorizontalAlignment', 'center');
    end
    title(title_text)
  case 'bottom'
    map = colormap; im(1,:,:) = map; im(2,:,:) = map;
    sz = size(map, 1);
    image(mmin+(0:sz-1) * d_mm/(sz-1), 1:2, uint8(fix(im*255)));
    set(h_a, 'YTick', [], 'TickDir', 'in', 'Box', 'on', 'YDir', 'normal');
    switch text_pos
      case 'none'
        set(h_a, 'XTickLabel', {}, 'TickLength', [.1,.1]);
    end
  otherwise
    map = the_colormap; im(:,1,:) = map; im(:,2,:) = map;
    sz = size(map, 1);
    image(1:2, mmin+(0:sz-1) * (mmax-mmin)/(sz-1), uint8(fix(im*255)), 'Parent', h_a);
    set(h_a, 'XTick', [], 'YTick', [], 'Box', 'on', 'YDir', 'normal');
    switch text_pos
      case 'center'
        im_ticks = calc_ticks(mmin, mmax);
        hold(h_a, 'on');
        set(h_a, 'YTick', [], 'YTickLabel', []);
        mark = 0.2;
        for ii=1:length(im_ticks)
          plot(0.5 + [0,mark], im_ticks(ii)*[1,1],'k')
          plot(2.5 + [-mark, 0], im_ticks(ii)*[1,1],'k')
          text(1.5, im_ticks(ii), sprintf('%g',im_ticks(ii)), 'HorizontalAlignment', 'center');
        end
        title(title_text)
      case 'right'
        set(h_a, 'TickDir', 'in', 'YAxisLocation', 'right');
        title(title_text)
    end
    axis(h_a, 'tight');
end
%--------------------------------------------------------------------------

function pos = axis_position(fig, type, args)

switch type
  case 'fixed'
    pos = get(fig, 'position');
  case 'manual'
    pos = safeget(args, 'position', [0.94, .1, 0.04, .8]);
    switch safeget(args, 'text', 'center')
      case 'center'
      case 'right', pos = pos - [0, 0, pos(3)/2, 0];
      case 'left', pos = pos + [pos(3)/2, 0, -pos(3)/2, 0];
    end
  case 'top'
    if isfield(args, 'position'), pos = args.position; return; end
    bar_left = safeget(args, 'bottom', 0.1);
    bar_bottom = safeget(args, 'top', 0.88);
    
    axs = findobj(fig, 'Type', 'axes');
    axs_pos = get(axs, 'Position'); axs_pos_max = [];
    for ii=1:size(axs_pos), kk = axs_pos{ii}; axs_pos_max(ii) = kk(2)+kk(4); end
    bar_top = max(axs_pos_max) + 0.005;
    
    pos = [bar_left, bar_bottom, 0.99 - bar_left, bar_top - bar_bottom-0.04];
    
  case 'right'
    if isfield(args, 'position'), pos = args.position; return; end
    a_bottom = safeget(args, 'bottom', 0.1);
    a_top = safeget(args, 'top', 0.88);
    
    axs = findobj(fig, 'Type', 'axes');
    axs_pos = get(axs, 'Position'); axs_pos_max = [];
    if ~iscell(axs_pos), axs_pos = {axs_pos}; end
    for ii=1:size(axs_pos), kk = axs_pos{ii}; axs_pos_max(ii) = kk(1)+kk(3); end
    bar_left = max(axs_pos_max) + 0.005;
    
    pos = [bar_left, a_bottom, 0.99 - bar_left, a_top - a_bottom-0.04];
    
    switch safeget(args, 'text', 'center')
      case 'center'
      case 'right', pos = pos - [0, 0, pos(3)/2, 0];
      case 'left', pos = pos + [pos(3)/2, 0, -pos(3)/2, 0];
    end
  otherwise
    if isfield(args, 'position'), pos = args.position; return; end
end
%--------------------------------------------------------------------------

function im_ticks = calc_ticks(mmin, mmax)
tick_step = (mmax - mmin) / 5;
tick_step = round(tick_step, 1, 'significant');

im_ticks = floor(mmin/tick_step)*tick_step : tick_step : floor(mmax/tick_step)*tick_step;
if length(im_ticks) < 4
  tick_step = tick_step / 2;
  im_ticks = floor(mmin/tick_step)*tick_step : tick_step : floor(mmax/tick_step)*tick_step;
end

im_ticks = im_ticks(im_ticks > mmin & im_ticks < mmax);
