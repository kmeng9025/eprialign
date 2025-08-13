function varargout = SetColorDLG(varargin)
% SETCOLORDLG M-file for SetColorDLG.fig
%      SETCOLORDLG, by itself, creates a new SETCOLORDLG or raises the existing
%      singleton*.
%
%      H = SETCOLORDLG returns the handle to a new SETCOLORDLG or the handle to
%      the existing singleton*.
%
%      SETCOLORDLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SETCOLORDLG.M with the given input arguments.
%
%      SETCOLORDLG('Property','Value',...) creates a new SETCOLORDLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SetColorDLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SetColorDLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SetColorDLG

% Last Modified by GUIDE v2.5 03-Feb-2011 07:28:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SetColorDLG_OpeningFcn, ...
                   'gui_OutputFcn',  @SetColorDLG_OutputFcn, ...
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


% --------------------------------------------------------------------
function SetColorDLG_OpeningFcn(hObject, eventdata, handles, varargin)
color = varargin{1};

handles.output.Color = safeget(color, 'Color', 'blue');
handles.output.Color2 = safeget(color, 'Color2', [1,0,0]);
handles.output.LineWidth = safeget(color, 'LineWidth', 0.5);
handles.output.FaceColor = safeget(color, 'FaceColor', 'blue');
handles.output.EdgeColor = safeget(color, 'EdgeColor', 'blue');
handles.output.FaceAlpha = safeget(color, 'FaceAlpha', 1);
handles.output.FaceVertexCData = safeget(color, 'FaceVertexCData', 'blue');
handles.output.FaceLighting = safeget(color, 'FaceLighting', 'flat');
handles.output.BackFaceLighting = safeget(color, 'BackFaceLighting', 'unlit');
handles.output.ContourThreshold = safeget(color, 'ContourThreshold', 0.5);
handles.output.LowCutOff = safeget(color, 'LowCutOff', 0.05);
handles.output.HighCutOff = safeget(color, 'HighCutOff', 0.95);
handles.output.CutOffAbsoluteScale = safeget(color, 'CutOffAbsoluteScale', 0);
handles.output.ColormapName = safeget(color, 'ColormapName', 'jet');
handles.output.MaskBelowThreshold = safeget(color, 'MaskBelowThreshold', 0);
handles.output.ShowColorbar = safeget(color, 'ShowColorbar', 0);
handles.output.SliceErode = fix(safeget(color, 'SliceErode', 0));
handles.output.SliceErode = max(handles.output.SliceErode, 0);
handles.output.SliceErode = min(handles.output.SliceErode, 4);
handles.output.SliceLargest = safeget(color, 'SliceLargest', 0);
handles.output.PostProcessing = safeget(color, 'PostProcessing', '');

% 
str = get(handles.pmLineWidth, 'String');
val = [];
for ii=1:length(str), val(end+1) = str2double(str{ii}); end
[a, idx] = min(abs(val - handles.output.LineWidth));
set(handles.pmLineWidth, 'Value', idx);
% 
set(handles.tColor, 'BackgroundColor', epr_rgb(handles.output.Color))
set(handles.tColor, 'ForegroundColor', 1-epr_rgb(handles.output.Color))
% 
if ischar(handles.output.FaceColor) && strcmp(handles.output.FaceColor, 'none')
  set(handles.cbUseFaceColor, 'Value', 0);
else
  set(handles.cbUseFaceColor, 'Value', 1);
  set(handles.tFaceColor, 'BackgroundColor', epr_rgb(handles.output.FaceColor))
  set(handles.tFaceColor, 'ForegroundColor', 1-epr_rgb(handles.output.FaceColor))
end
% 
if ischar(handles.output.EdgeColor) && strcmp(handles.output.EdgeColor, 'none')
  set(handles.cbUseEdgeColor, 'Value', 0);
else
  set(handles.cbUseEdgeColor, 'Value', 1);
  set(handles.tEdgeColor, 'BackgroundColor', epr_rgb(handles.output.EdgeColor))
  set(handles.tEdgeColor, 'ForegroundColor', 1-epr_rgb(handles.output.EdgeColor))
end
% 
str = get(handles.pmFaceAlpha, 'String');
val = [];
for ii=1:length(str), val(end+1) = str2double(str{ii}); end
[a, idx] = min(abs(val - handles.output.FaceAlpha));
set(handles.pmFaceAlpha, 'Value', idx);
% 
str = get(handles.pmFacelighting, 'String');
for ii=1:length(str), 
  if strcmp(str{ii}, handles.output.FaceLighting)
    set(handles.pmFacelighting, 'Value', ii); break;
  end
end
%
set(handles.ePostprocessing, 'String', handles.output.PostProcessing)
%
if numel(handles.output.ContourThreshold) > 1
    str = sprintf('%g,',handles.output.ContourThreshold);
    pos = find(str == ',', 1, 'last');
    if ~isempty(pos), str = ['[',str(1:pos-1), ']']; end
    set(handles.eContourThreshold, 'String', str);    
else
    set(handles.eContourThreshold, 'String', num2str(handles.output.ContourThreshold));
end
set(handles.eLowCutOff, 'String', num2str(handles.output.LowCutOff));
set(handles.eHighCutOff, 'String', num2str(handles.output.HighCutOff));
set(handles.cbAbsoluteScale, 'Value', handles.output.CutOffAbsoluteScale)

str = get(handles.pmColormap, 'String');
for ii=1:length(str)
  if isequal(str{ii}, handles.output.ColormapName)
    set(handles.pmColormap, 'Value', ii);
  end
  break;
end
set(handles.cbShowColorBar, 'Value', handles.output.ShowColorbar);
set(handles.cbMaskData, 'Value', handles.output.MaskBelowThreshold);
set(handles.pmSliceErode, 'Value', handles.output.SliceErode + 1);
set(handles.cbSliceLargest, 'Value', handles.output.SliceLargest);

% Update handles structure
guidata(hObject, handles);
uiwait(handles.figure1);

% --------------------------------------------------------------------
function varargout = SetColorDLG_OutputFcn(hObject, eventdata, handles) 
% 
val = get(handles.pmLineWidth, 'Value');
str = get(handles.pmLineWidth, 'String');
handles.output.LineWidth = str2double(str{val});
% 
if get(handles.cbUseFaceColor, 'Value')
else
  handles.output.FaceColor = 'none';
end
% 
if get(handles.cbUseEdgeColor, 'Value')
else
  handles.output.EdgeColor = 'none';
end
% 
val = get(handles.pmFaceAlpha, 'Value');
str = get(handles.pmFaceAlpha, 'String');
handles.output.FaceAlpha = str2double(str{val});
% 
val = get(handles.pmFacelighting, 'Value');
str = get(handles.pmFacelighting, 'String');
handles.output.FaceLighting = str{val};

handles.output.ContourThreshold = eval(get(handles.eContourThreshold, 'String'));
handles.output.LowCutOff = str2double(get(handles.eLowCutOff, 'String'));
handles.output.HighCutOff = str2double(get(handles.eHighCutOff, 'String'));
handles.output.CutOffAbsoluteScale = get(handles.cbAbsoluteScale, 'Value');

val = get(handles.pmColormap, 'Value');
str = get(handles.pmColormap, 'String');
handles.output.ColormapName = str{val}; 
handles.output.MaskBelowThreshold = get(handles.cbMaskData, 'Value');
handles.output.ShowColorbar = get(handles.cbShowColorBar, 'Value');
handles.output.SliceErode = get(handles.pmSliceErode, 'Value')-1;
handles.output.SliceLargest = get(handles.cbSliceLargest, 'Value');
handles.output.PostProcessing = get(handles.ePostprocessing, 'String');

varargout{1} = handles.output;
delete(hObject);

% --------------------------------------------------------------------
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% Disable the close request using system menu

% --------------------------------------------------------------------
function pbColor_Callback(hObject, eventdata, handles)
cc = uisetcolor;

if length(cc) == 3
  handles.output.Color = cc;
  set(handles.tColor, 'BackgroundColor', handles.output.Color)
  set(handles.tColor, 'ForegroundColor', 1-handles.output.Color)
  guidata(hObject, handles);
end

% --------------------------------------------------------------------
function pbColor2_Callback(hObject, eventdata, handles)
cc = uisetcolor;

if length(cc) == 3
  handles.output.Color2 = cc;
  guidata(hObject, handles);
end

% --------------------------------------------------------------------
function pbDone_Callback(hObject, eventdata, handles)
uiresume(gcbf);

% --------------------------------------------------------------------
function pbFaceColor_Callback(hObject, eventdata, handles)
cc = uisetcolor;

if length(cc) == 3
  handles.output.FaceColor = cc;
  set(handles.tFaceColor, 'BackgroundColor', handles.output.FaceColor)
  set(handles.tFaceColor, 'ForegroundColor', 1-handles.output.FaceColor)
  guidata(hObject, handles);
end

% --------------------------------------------------------------------
function pbEdgeColor_Callback(hObject, eventdata, handles)
cc = uisetcolor;

if length(cc) == 3
  handles.output.EdgeColor = cc;
  set(handles.tEdgeColor, 'BackgroundColor', handles.output.EdgeColor)
  set(handles.tEdgeColor, 'ForegroundColor', 1-handles.output.EdgeColor)
  guidata(hObject, handles);
end

% --------------------------------------------------------------------
function pbVertexData_Callback(hObject, eventdata, handles)
