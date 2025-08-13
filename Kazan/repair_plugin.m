function varargout = repair_plugin(varargin)
% REPAIR_PLUGIN M-file for repair_plugin.fig
%      REPAIR_PLUGIN, by itself, creates a new REPAIR_PLUGIN or raises the existing
%      singleton*.
%
%      H = REPAIR_PLUGIN returns the handle to a new REPAIR_PLUGIN or the handle to
%      the existing singleton*.
%
%      REPAIR_PLUGIN('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in REPAIR_PLUGIN.M with the given input arguments.
%
%      REPAIR_PLUGIN('Property','Value',...) creates a new REPAIR_PLUGIN or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before repair_plugin_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to repair_plugin_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help repair_plugin

% Last Modified by GUIDE v2.5 27-Mar-2008 18:24:50

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @repair_plugin_OpeningFcn, ...
                   'gui_OutputFcn',  @repair_plugin_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before repair_plugin is made visible.
function repair_plugin_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;

[fpath] = fileparts(which('kazan'));
inifilename = [fpath '\kazan.ini'];
ini = inimanage(inifilename);

handles.Repair.PTS = [];
handles.Repair.Window = [];

% Update handles structure
guidata(hObject, handles);


% --- Outputs from this function are returned to the command line.
function varargout = repair_plugin_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;


function figure1_DeleteFcn(hObject, eventdata, handles)

if isfield(handles, 'hh')
  [~,name] = fileparts(get(handles.hh, 'FileName'));
  eval([name '(''DeletePlugins'', handles.figure1, handles.hh)']);
end

function FileUpdate(hObject, handles)

function SelectionUpdate(hObject, par, handles)

function lbPoints_Callback(hObject, eventdata, handles)

function slider1_Callback(hObject, eventdata, handles)


% --------------------------------------------------------------------
function varargout = Plot(handles)
[path,name,ext] = fileparts(get(handles.hh, 'FileName'));

hand = guidata(handles.hh);
hand.out = {};
tt = hand.src;
tt.title = 'Data';

if get(handles.cbUseMirrorCopy, 'Value'), useMirror = 1; else useMirror = 0; end
if get(handles.rbMirrorCopy, 'Value'), useMirror = 1; end
  
if useMirror
    % show slice which is symmetric to the shown
  prj_num = fix((hand.Selection-1)/14);
  sel_our = (hand.Selection-1) - prj_num*14;
  sel_mirror = prj_num*14 + ((14-1)-sel_our) + 1;
    switch hand.Projection
      case 1 % X
        mirror = tt.y(:, sel_mirror)';
      case 2 % Y
        mirror = tt.y(sel_mirror, :);
    end
   mirror = -mirror(end:-1:1);
   shift = str2num(get(handles.eShift, 'String'));
   mirror = spline(-9:length(mirror)+10, [zeros(1,10),mirror,zeros(1,10)], [1:length(mirror)]+shift)';
end

if get(handles.rbSplineFix, 'Value')
  % repair procedure
  if ~isempty(handles.Repair.Window) & ~isempty(handles.Repair.PTS)
    switch hand.Projection
      case 1 % X
        % old data
        old_y = tt.y(handles.Repair.Window, hand.Selection);
        old_x = tt.ax.x(handles.Repair.Window);
        new_y = spline(handles.Repair.PTS(:,1), ...
          handles.Repair.PTS(:,2),old_x);
        tt.y(handles.Repair.Window, hand.Selection) = new_y;
        hand.out{1} = tt;
      case 2 % Y
        hand.out{1} = tt;
    end
    
    if useMirror
      slice.y    = mirror;
      slice.ax.x = tt.ax.x;
      % spl.ax.Marker    = '*';
      slice.ax.Color     = 'm';
      % spl.ax.LineStyle = 'none';
      slice.title = 'Mirror';
      hand.out{end+1} = slice;
    end
    
    old.y    = old_y;
    old.ax.x = old_x;
    old.ax.Color   = 'k';
    old.title = 'Old';
    hand.out{end+1} = old;
    new.y    = new_y;
    new.ax.x = old_x;
    new.ax.Color   = 'b';
    new.ax.Marker  = '.';
    new.ax.LineStyle = 'none';
    new.title = 'Old';
    hand.out{end+1} = new;
  else
    hand.out{1} = tt;
    
    if useMirror
      slice.y    = mirror;
      slice.ax.x = tt.ax.x;
      % spl.ax.Marker    = '*';
      slice.ax.Color     = 'm';
      % spl.ax.LineStyle = 'none';
      slice.title = 'Mirror';
      hand.out{end+1} = slice;
    end
  end

  if size(handles.Repair.PTS,1) > 0
    % Show points
    spl.y    = handles.Repair.PTS(:,2);
    spl.ax.x = handles.Repair.PTS(:,1);
    spl.ax.Marker    = '*';
    spl.ax.Color     = 'r';
    spl.ax.LineStyle = 'none';
    spl.title = 'Marker';
    hand.out{end+1} = spl;

    % show spline
    spl1.y    = spline(handles.Repair.PTS(:,1), ...
      handles.Repair.PTS(:,2),tt.ax.x);
    spl1.ax.x = tt.ax.x;
    spl1.ax.Color     = 'r';
    spl1.title = 'Spline';
    hand.out{end+1} = spl1;
  end
elseif get(handles.rbMirrorCopy, 'Value')
  % repair procedure
  if ~isempty(handles.Repair.Window)
    switch hand.Projection
      case 1 % X
        % old data
        old_y = tt.y(handles.Repair.Window, hand.Selection);
        old_x = tt.ax.x(handles.Repair.Window);
        new_y = mirror(handles.Repair.Window);
        tt.y(handles.Repair.Window, hand.Selection) = new_y;
        hand.out{end+1} = tt;
      case 2 % Y
        hand.out{end+1} = tt;
    end

    if useMirror
      slice.y    = mirror;
      slice.ax.x = tt.ax.x;
      % spl.ax.Marker    = '*';
      slice.ax.Color     = 'm';
      % spl.ax.LineStyle = 'none';
      slice.title = 'Mirror';
      hand.out{end+1} = slice;
    end

    old.y    = old_y;
    old.ax.x = old_x;
    old.ax.Color     = 'k';
    old.title = 'Old';
    hand.out{end+1} = old;
    new.y    = new_y;
    new.ax.x = old_x;
    new.ax.Color   = 'b';
    new.ax.Marker  = '.';
    new.ax.LineStyle = 'none';
    new.title = 'Old';
    hand.out{end+1} = new;
  else
    hand.out{1} = tt;
    
    if useMirror
      slice.y    = mirror;
      slice.ax.x = tt.ax.x;
      % spl.ax.Marker    = '*';
      slice.ax.Color     = 'm';
      % spl.ax.LineStyle = 'none';
      slice.title = 'Mirror';
      hand.out{end+1} = slice;
    end
  end
end

[~,name] = fileparts(get(handles.hh, 'FileName'));
eval([name '(''SetDataSource'', 2, hand)']);

% --------------------------------------------------------------------
function PointsList(handles)
Str = {};

for ii=1:size(handles.Repair.PTS, 1)
    Str{end+1} = sprintf('%-8.3f  %-8.3f',handles.Repair.PTS(ii,1),...
      handles.Repair.PTS(ii,2));
end

set(handles.lbPoints,'String',Str);

% --------------------------------------------------------------------
function pbAddPoint_Callback(hObject, eventdata, handles)
hand = guidata(handles.hh);
[~,name] = fileparts(get(handles.hh, 'FileName'));
[x,y]=eval([name '(''GetPoint'', hand, 88888)']);

axx = hand.src.ax.x;

for ii=1:length(x)
    handles.Repair.PTS(end+1, 1) = x(ii);
    handles.Repair.PTS(end, 2)   = y(ii);
end

[zzz, idx] = sort( handles.Repair.PTS(:, 1));
handles.Repair.PTS(idx, 1) = handles.Repair.PTS(:, 1);
handles.Repair.PTS(idx, 2) = handles.Repair.PTS(:, 2);
guidata(handles.figure1, handles);

PointsList(handles);
Plot(handles)

% --------------------------------------------------------------------
function pbDeletePoint_Callback(hObject, eventdata, handles)

hand = guidata(handles.hh);
[~,name] = fileparts(get(handles.hh, 'FileName'));
[x,y]=eval([name '(''GetPoint'', hand, 88888)']);
if ~length(x), return; end

for ii=1:length(x)
  gidx = 1:size(handles.Repair.PTS, 1);
  % find point to delete
  [mm, idx] = min(abs(handles.Repair.PTS(:,1)-x(ii)));
  gidx = gidx(gidx~=idx);
  
  handles.Repair.PTS = handles.Repair.PTS(gidx,:);
end
guidata(handles.figure1, handles);

PointsList(handles);
Plot(handles)

% --------------------------------------------------------------------
function pbAdjustPoint_Callback(hObject, eventdata, handles)
% find point
hand = guidata(handles.hh);
[~,name] = fileparts(get(handles.hh, 'FileName'));
[x,y]=eval([name '(''GetPoint'', hand, 888)']);
if ~length(x), return; end
[xx,yy] = eval([name '(''GetOnScreen1DSlice'', hand)']);
if ~length(xx), return; end
yy = smooth(yy,5);

for ii=1:length(x)
  % find point to correct
  [mm, idx] = min(abs(handles.Repair.PTS(:,1)-x(ii)));
  % find data point mostly closed
  [mm, idx_yy] = min(abs(xx-x(ii)));
  % correct point
  handles.Repair.PTS(idx,2) = yy(idx_yy);
end

guidata(handles.figure1, handles);

PointsList(handles);
Plot(handles)

% --------------------------------------------------------------------
function pbAdjustAllPoint_Callback(hObject, eventdata, handles)
% find point
hand = guidata(handles.hh);
[~,name] = fileparts(get(handles.hh, 'FileName'));
[xx,yy] = eval([name '(''GetOnScreen1DSlice'', hand)']);
if ~length(xx), return; end
yy = smooth(yy,5);

for ii=1:size(handles.Repair.PTS, 1)
  % find data point mostly closed
  [mm, idx_yy] = min(abs(xx-handles.Repair.PTS(ii,1)));
  % correct point
  handles.Repair.PTS(ii,2) = yy(idx_yy);
end

guidata(handles.figure1, handles);

PointsList(handles);
Plot(handles)

% --------------------------------------------------------------------
function pbDeleteAllPoint_Callback(hObject, eventdata, handles)
handles.Repair.PTS = [];
guidata(handles.figure1, handles);

PointsList(handles);
Plot(handles)

% --------------------------------------------------------------------
function AddAllPoints_Callback(hObject, eventdata, handles)
% find point
hand = guidata(handles.hh);
[~,name] = fileparts(get(handles.hh, 'FileName'));
[xx,yy] = eval([name '(''GetOnScreen1DSlice'', hand)']);
if ~length(xx), return; end
yy = smooth(yy,5);

dx = xx(2)-xx(1);
handles.Repair.PTS = [];
handles.Repair.PTS(:,1) = xx(1)- 3*dx : 10*dx: xx(end)+ 3*dx;
handles.Repair.PTS(:,2) = 0;

guidata(handles.figure1, handles);
pbAdjustAllPoint_Callback(hObject, eventdata, handles);

% --------------------------------------------------------------------
function pbChangePoint_Callback(hObject, eventdata, handles)

while 1
  % find point
  hand = guidata(handles.hh);
  [~,name] = fileparts(get(handles.hh, 'FileName'));
  [x,y]=eval([name '(''GetPoint'', hand, 2)']);
  if length(x)~=2, return; end

  [mm, idx] = min(abs(x(1)-handles.Repair.PTS(:,1)));
  handles.Repair.PTS(idx,1) = x(2);
  handles.Repair.PTS(idx,2) = y(2);
  guidata(handles.figure1, handles);

  PointsList(handles);
  Plot(handles)
end

% --------------------------------------------------------------------
function WindowList(handles)
Str = [];
if isempty(handles.Repair.Window)
  Str = '[]';
else
  Str = num2str(handles.Repair.Window(1));
  for ii=2:length(handles.Repair.Window)
    Str = [Str,',',num2str(handles.Repair.Window(ii))];
  end
end
set(handles.eWindow,'String',Str);

% --------------------------------------------------------------------
function pbAddWindow_Callback(hObject, eventdata, handles)
% get window
hand = guidata(handles.hh);
[~,name] = fileparts(get(handles.hh, 'FileName'));
[x,y]=eval([name '(''GetPoint'', hand, 2)']);
if length(x)~=2, return; end
[xx,yy] = eval([name '(''GetOnScreen1DSlice'', hand)']);
if ~length(xx), return; end

idx = 1:length(xx)';

handles.Repair.Window = idx(xx(idx) >= x(1) & xx(idx) <= x(2));
guidata(handles.figure1, handles);

WindowList(handles);
Plot(handles)

% --------------------------------------------------------------------
function pbRemoveWindow_Callback(hObject, eventdata, handles)
handles.Repair.Window = [];
guidata(handles.figure1, handles);

WindowList(handles);
Plot(handles)


% --------------------------------------------------------------------
function pbRefresh_Callback(hObject, eventdata, handles)

% search for zero
if get(handles.cbUseMirrorCopy, 'Value'), useMirror = 1; else useMirror = 0; end
if get(handles.rbMirrorCopy, 'Value'), useMirror = 1; end
  
if useMirror
  hand = guidata(handles.hh);
  hand.out = {};
  tt = hand.src;
    % show slice which is symmetric to the shown
  prj_num = fix((hand.Selection-1)/14);
  sel_our = (hand.Selection-1) - prj_num*14;
  sel_mirror = prj_num*14 + ((14-1)-sel_our) + 1;
    switch hand.Projection
      case 1 % X
        src_trace = tt.y(:, hand.Selection)';
        mirror = tt.y(:, sel_mirror)';
      case 2 % Y
        mirror = tt.y(sel_mirror, :);
    end
   mirror = -mirror(end:-1:1);
   shift = str2num(get(handles.eShift, 'String'));
   
   shift_it = @(x, yy) ...
      spline(-9:length(yy)+10, [zeros(1,10),yy,zeros(1,10)], [1:length(yy)]+x);
   shift_it_diff = @(x, yy, yy1) sqrt(sum((yy - shift_it(x, yy1)).^2)); 
   XX = fminsearch(shift_it_diff, shift, [], src_trace, mirror);
   set(handles.eShift, 'String', num2str(fix(XX*100)/100));
end


PointsList(handles);
Plot(handles)

% --------------------------------------------------------------------
function pbSave_Callback(hObject, eventdata, handles)
hand = guidata(handles.hh);
tt = hand.src;

[FileName,PathName] = uiputfile({'*.img', 'Image file (*.img)'},'Save file');
if PathName ~= 0
  if get(handles.rbLabview, 'Value')
    WriteCWImageFile(fullfile(PathName,FileName), tt.y, tt.dsc, 'Format', 'Labview')
  else
    WriteCWImageFile(fullfile(PathName,FileName), tt.y, tt.dsc, 'Format', 'Modula')
  end
end


% --------------------------------------------------------------------
function cbUseMirrorCopy_Callback(hObject, eventdata, handles)


% --------------------------------------------------------------------
function eShift_Callback(hObject, eventdata, handles)

Plot(handles)
% --------------------------------------------------------------------
function slShift_Callback(hObject, eventdata, handles)

shift = get(handles.slShift, 'Value');
set(handles.slShift, 'Value',0);

val = str2num(get(handles.eShift, 'String'));
set(handles.eShift, 'String', num2str(val + 0.1*shift))
Plot(handles)


% --------------------------------------------------------------------
function bboxMethod_SelectionChangeFcn(hObject, eventdata, handles)

Plot(handles)

% --------------------------------------------------------------------
function pbAccept_Callback(hObject, eventdata, handles)

[path,name,ext] = fileparts(get(handles.hh, 'FileName'));

hand = guidata(handles.hh);
hand.src = hand.out{1};
guidata(handles.hh, hand);

Plot(handles)

% --------------------------------------------------------------------
function pbMirrorCopyAutoFixFlyback_Callback(hObject, eventdata, handles)

if get(handles.cbUseMirrorCopy, 'Value'), useMirror = 1; else useMirror = 0; end
if isempty(handles.Repair.Window), 
  useMirror = 0; 
  disp('No window selected for the repair procedure.');
end

if useMirror
  % search for zero
  hand = guidata(handles.hh);
  hand.out = {};
  tt = hand.src;
  hand.out{1} = tt;
  hand.out{1}.title = 'Source';

  % prepare the fitting function
  shift_it = @(x, yy) ...
    spline(-9:length(yy)+10, [zeros(1,10),yy,zeros(1,10)], [1:length(yy)]+x);
  shift_it_diff = @(x, yy, yy1) sqrt(sum((yy - shift_it(x, yy1)).^2));
  shift = str2num(get(handles.eShift, 'String'));

  % show slice which is symmetric to the shown
  switch hand.Projection
    case 1 % X
      for ii=1:size(tt.y,2)
        prj_num = fix((ii-1)/14);
        sel_our = (ii-1) - prj_num*14;
        sel_mirror = prj_num*14 + ((14-1)-sel_our) + 1;
        src_trace = tt.y(:, ii)';
        mirror = -tt.y(end:-1:1, sel_mirror)';

        % find shift
        XX = fminsearch(shift_it_diff, shift, [], src_trace, mirror);
        
        mirror = shift_it(XX, mirror);
        tt.y(handles.Repair.Window, ii) = mirror(handles.Repair.Window);
      end
    case 2 % Y
      mirror = tt.y(sel_mirror, :);
  end
  disp('Auto flyback corection is finished.');
  
  hand.out{2} = tt;
  hand.out{2}.ax.Color     = 'm';
  hand.out{2}.title = 'Fixed';

  [~,name] = fileparts(get(handles.hh, 'FileName'));
  eval([name '(''SetDataSource'', 2, hand)']);
end

% --- Executes on button press in pbModifyXScale.
function pbModifyXScale_Callback(hObject, eventdata, handles)

xcor = str2double(get(handles.eXScale, 'String'));

hand = guidata(handles.hh);
hand.out = {};
tt = hand.src;
hand.out{1} = tt;
hand.out{1}.title = 'Source';

sz = size(tt.y);
x = 1:sz(1); x1 = x * xcor + (sz(1)+1)*(1-xcor)/2;
for ii=1:sz(2)
  tt.y(:,ii) = spline(x1, tt.y(:,ii), x);
end

hand.out{2} = tt;
hand.out{2}.ax.Color     = 'm';
hand.out{2}.title = 'Fixed';

[~,name] = fileparts(get(handles.hh, 'FileName'));
eval([name '(''SetDataSource'', 2, hand)']);


