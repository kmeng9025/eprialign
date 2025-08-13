% Rotation toolbar.
% create:
%     FigureNavigationPanelPNL
%     FigureNavigationPanelPNL('create')
%     FigureNavigationPanelPNL('create', figure_number)
% destroy:
%     FigureNavigationPanelPNL('destroy', figure_number)

function FigureNavigationPanelPNL(command_line, varargin)

if nargin == 0, command_line = 'create'; end

switch lower(command_line)
  case 'create'
    if nargin < 2, hfig = gcf; else  hfig = varargin{1}; end
    
    obj = findobj(hfig, 'Type', 'uipanel');
    if ~isempty(obj), return; end

    hh.panel = uipanel('Title', 'Navigation (camera)', 'Unit', 'pixels', 'Parent', hfig);
    
    hh.buttonUP = uicontrol('Style', 'pushbutton', 'Parent', hh.panel, ...
      'String', '^', 'Units', 'pixels', ...
      'Callback', sprintf('FigureNavigationPanelPNL(''UP'',%i)', hfig));
    hh.buttonDN = uicontrol('Style', 'pushbutton', 'Parent', hh.panel, ...
      'String', 'v', 'Units', 'pixels', ...
      'Callback', sprintf('FigureNavigationPanelPNL(''DN'',%i)', hfig));
    hh.buttonINPL1 = uicontrol('Style', 'pushbutton', 'Parent', hh.panel, ...
      'String', '^-', 'Units', 'pixels', ...
      'Callback', sprintf('FigureNavigationPanelPNL(''INPL1'',%i)', hfig));
    hh.buttonINPL2 = uicontrol('Style', 'pushbutton', 'Parent', hh.panel, ...
      'String', '-v', 'Units', 'pixels', ...
      'Callback', sprintf('FigureNavigationPanelPNL(''INPL2'',%i)', hfig));
    hh.buttonOUTPL1 = uicontrol('Style', 'pushbutton', 'Parent', hh.panel, ...
      'String', '<-', 'Units', 'pixels', ...
      'Callback', sprintf('FigureNavigationPanelPNL(''OUTPL1'',%i)', hfig));
    hh.buttonOUTPL2 = uicontrol('Style', 'pushbutton', 'Parent', hh.panel, ...
      'String', '->', 'Units', 'pixels', ...
      'Callback', sprintf('FigureNavigationPanelPNL(''OUTPL2'',%i)', hfig));
    hh.pmAngle = uicontrol('Style', 'popupmenu', 'Parent', hh.panel, ...
      'String', {'90';'45';'25';'15';'5';'1'}, 'Value', 1, 'Units', 'pixels');
    set(hh.panel, 'UserData', hh)
    set(hfig, 'ResizeFcn', sprintf('FigureNavigationPanelPNL(''resize'',%i)', hfig));
    eval(sprintf('FigureNavigationPanelPNL(''resize'',%i)', hfig));
  case 'destroy'
    hfig = varargin{1};
    obj = findobj(hfig, 'Type', 'uipanel');
    hh = get(obj, 'UserData');
    if ~isempty(hh.buttonINPL1), delete(hh.buttonINPL1); end
    if ~isempty(hh.buttonINPL2), delete(hh.buttonINPL2); end
    if ~isempty(hh.buttonOUTPL1), delete(hh.buttonOUTPL1); end
    if ~isempty(hh.buttonOUTPL2), delete(hh.buttonOUTPL2); end
    if ~isempty(hh.buttonUP), delete(hh.buttonUP); end
    if ~isempty(hh.buttonDN), delete(hh.buttonDN); end
    if ~isempty(hh.pmAngle), delete(hh.pmAngle); end
    if ~isempty(hh.panel), delete(hh.panel); end
    set(hfig, 'ResizeFcn', '');
  case 'resize'
    hfig = varargin{1};
    fig_dim = get(hfig, 'Position');
    obj = findobj(hfig, 'Type', 'uipanel');
    hh = get(obj, 'UserData');
    btn_size = 31; btn_space = 3;
    h_panel = 3*btn_space+2*btn_size+12; w_panel = 5*btn_space+4*btn_size+3;
    
    if ~isempty(obj)
      set(hh.panel, 'Position', [fig_dim(3)-w_panel-3, 3, w_panel, h_panel-3]);
      set(hh.buttonINPL1, 'Position', [btn_space, btn_space, btn_size, btn_size]);
      set(hh.buttonINPL2, 'Position', [2*btn_space+btn_size, btn_space, btn_size, btn_size]);
      set(hh.buttonOUTPL1, 'Position', [3*btn_space+2*btn_size, btn_space, btn_size, btn_size]);
      set(hh.buttonOUTPL2, 'Position', [4*btn_space+3*btn_size, btn_space, btn_size, btn_size]);
      set(hh.buttonUP, 'Position', [btn_space, btn_space+btn_size, btn_size, btn_size]);
      set(hh.buttonDN, 'Position', [2*btn_space+btn_size, btn_space+btn_size, btn_size, btn_size]);
      set(hh.pmAngle,  'Position', [3*btn_space+2*btn_size+2, btn_space+btn_size, 2*btn_size-4, btn_size - 4]);
    end
  case 'up'
    hfig  = varargin{1};
    angle = get_angle(hfig);
    ax = findobj(hfig, 'Type', 'axes');
    CameraTarget = get(ax, 'CameraTarget');
    CameraPosition = get(ax, 'CameraPosition');
    CameraUpVector = get(ax, 'CameraUpVector');
    vec = cross(CameraPosition-CameraTarget, CameraUpVector);
    rotation = hmatrix_rotate_about(vec, +angle);
    set(ax, 'CameraUpVector', CameraUpVector*rotation(1:3,1:3), ...
      'CameraPosition',  CameraTarget+(CameraPosition-CameraTarget)*rotation(1:3,1:3))
  case 'dn'
    hfig  = varargin{1};
    angle = get_angle(hfig);
    ax = findobj(hfig, 'Type', 'axes');
    CameraTarget = get(ax, 'CameraTarget');
    CameraPosition = get(ax, 'CameraPosition');
    CameraUpVector = get(ax, 'CameraUpVector');
    vec = cross(CameraPosition-CameraTarget, CameraUpVector);
    rotation = hmatrix_rotate_about(vec, -angle);
    set(ax, 'CameraUpVector', CameraUpVector*rotation(1:3,1:3), ...
      'CameraPosition',  CameraTarget+(CameraPosition-CameraTarget)*rotation(1:3,1:3))
  case 'inpl1'
    hfig  = varargin{1};
    angle = get_angle(hfig);
    ax = findobj(hfig, 'Type', 'axes');
    CameraTarget = get(ax, 'CameraTarget');
    CameraPosition = get(ax, 'CameraPosition');
    CameraUpVector = get(ax, 'CameraUpVector');
    rotation = hmatrix_rotate_about(CameraPosition-CameraTarget, -angle);
    set(ax, 'CameraUpVector', CameraUpVector*rotation(1:3,1:3))
  case 'inpl2'
    hfig  = varargin{1};
    angle = get_angle(hfig);
    ax = findobj(hfig, 'Type', 'axes');
    CameraTarget = get(ax, 'CameraTarget');
    CameraPosition = get(ax, 'CameraPosition');
    CameraUpVector = get(ax, 'CameraUpVector');
    rotation = hmatrix_rotate_about(CameraPosition-CameraTarget, angle);
    set(ax, 'CameraUpVector', CameraUpVector*rotation(1:3,1:3))
  case 'outpl1'
    hfig  = varargin{1};
    angle = get_angle(hfig);
    ax = findobj(hfig, 'Type', 'axes');
    CameraTarget = get(ax, 'CameraTarget');
    CameraPosition = get(ax, 'CameraPosition');
    CameraUpVector = get(ax, 'CameraUpVector');
    rotation = hmatrix_rotate_about(CameraUpVector, -angle);
    set(ax, 'CameraPosition', CameraTarget+(CameraPosition-CameraTarget)*rotation(1:3,1:3))
  case 'outpl2'
    hfig  = varargin{1};
    angle = get_angle(hfig);
    ax = findobj(hfig, 'Type', 'axes');
    CameraTarget = get(ax, 'CameraTarget');
    CameraPosition = get(ax, 'CameraPosition');
    CameraUpVector = get(ax, 'CameraUpVector');
    rotation = hmatrix_rotate_about(CameraUpVector, angle);
    set(ax, 'CameraPosition', CameraTarget+(CameraPosition-CameraTarget)*rotation(1:3,1:3))
end

function angle = get_angle(hfig)
obj = findobj(hfig, 'Type', 'uipanel');
hh = get(obj, 'UserData');

String = get(hh.pmAngle, 'String');
Value = get(hh.pmAngle, 'Value');
angle = str2double(String{Value});
