function varargout = MatrixGUI(varargin)
% MATRIXGUI MATLAB code for MatrixGUI.fig
%      MATRIXGUI, by itself, creates a new MATRIXGUI or raises the existing
%      singleton*.
%
%      H = MATRIXGUI returns the handle to a new MATRIXGUI or the handle to
%      the existing singleton*.
%
%      MATRIXGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MATRIXGUI.M with the given input arguments.
%
%      MATRIXGUI('Property','Value',...) creates a new MATRIXGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MatrixGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MatrixGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MatrixGUI

% Last Modified by GUIDE v2.5 04-Sep-2014 16:07:50

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MatrixGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @MatrixGUI_OutputFcn, ...
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

% -------------------------------------------------------------------------
function MatrixGUI_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;

kk = 1;
for ii=4:nargin
  if iscell(varargin{kk})
    % load data
    s1 = load(varargin{kk}{1});
    fnames = fieldnames(s1);
    if ~isempty(fnames)
      handles.matrix{kk}.data = s1.(fnames{1});
    end
  else
      handles.matrix{kk}.data = varargin{kk};
  end
  handles.matrix{kk}.sz = [size(handles.matrix{kk}.data),1,1];
  kk = kk + 1;
end
sz = handles.matrix{1}.sz;

szM = length(handles.matrix);
if szM > 1 && ~isequal(handles.matrix{2}.sz, sz)
  a2 = handles.matrix{2}.data;
  sz2 = handles.matrix{2}.sz;
  handles.matrix{2}.data = zeros(sz);
  for ii=1:sz(2),
    for jj=1:sz(3),
      for kk=1:sz(4)
        handles.matrix{2}.data(:,ii,jj,kk) = interp1(linspace(0,1,sz2(1)),a2(:,ii,jj,kk),linspace(0,1,sz(1)));
      end;
    end;
  end
end
hh = [handles.sl1, handles.sl2, handles.sl3, handles.sl4];

sz = handles.matrix{1}.sz;
for ii=1:4
  if sz(ii) > 1
    set(hh(ii), 'SliderStep', [1 / max(sz(ii)-1,1), 0.1],...
      'Max', sz(ii), 'Min', 1, 'Value', sz(ii));
  else
    set(hh(ii), 'SliderStep', [0.1, 0.1],...
      'Max', 1, 'Min', 0, 'Value', sz(ii), 'Visible','off');
  end
end

guidata(hObject, handles);
set(handles.cbAx1, 'value',1)
cbAx_Callback(handles.cbAx1, [], handles);
% UIWAIT makes MatrixGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% -------------------------------------------------------------------------
function varargout = MatrixGUI_OutputFcn(hObject, eventdata, handles) 

varargout{1} = handles.output;

% -------------------------------------------------------------------------
function sl2_Callback(hObject, eventdata, handles)
PlotIt(handles)

% -------------------------------------------------------------------------
function sl3_Callback(hObject, eventdata, handles)
PlotIt(handles)

% -------------------------------------------------------------------------
function sl4_Callback(hObject, eventdata, handles)
PlotIt(handles)

% -------------------------------------------------------------------------
function sl1_Callback(hObject, eventdata, handles)
PlotIt(handles)

% -------------------------------------------------------------------------
function PlotIt(handles)
hh = [handles.cbAx1, handles.cbAx2, handles.cbAx3, handles.cbAx4];
plot_dims = get(hh,'value');
p1 = fix(get(handles.sl1, 'Value')); p1 = max(p1,1);
p2 = fix(get(handles.sl2, 'Value')); p2 = max(p2,1);
p3 = fix(get(handles.sl3, 'Value')); p3 = max(p3,1);
p4 = fix(get(handles.sl4, 'Value')); p4 = max(p4,1);
szM = length(handles.matrix);
colorsR = {'b','r','k'};
colorsI = {'g','m','c'};

cla(handles.axes1); hold(handles.axes1, 'on');
try
  if plot_dims{1}
    set(handles.ePos,'string',sprintf('[: %i %i %i]',p2,p3,p4));
    maxY = zeros(szM, 1);
    d1 = cell(szM, 1);
    for ii=1:szM
      d1{ii} = handles.matrix{ii}.data(:,p2,p3,p4);
      maxY(ii) = max(real(d1{ii}));
    end
    if szM == 1, 
      maxY(1) = 1; 
    else
      maxY = maxY(1) ./ maxY;
    end
    for ii=1:szM
      plot(1:length(d1{ii}), real(squeeze(d1{ii})*maxY(ii)), ...
        colorsR{ii}, 'Parent', handles.axes1);
      if ~isreal(d1{ii})
        plot(1:length(d1{ii}), imag(squeeze(d1{ii})*maxY(ii)), ...
          colorsI{ii}, 'Parent', handles.axes1);
      end
      if ii > 1
        text(0.05, 0.9, sprintf('max1/max2 = %5.3f',maxY(ii)), 'units', 'normalized')
        text(0.05, 0.80, sprintf('integral 2 = %5.3f',sum(real(d1{ii}))), 'units', 'normalized')
      end
    end
    axis(handles.axes1, 'tight');
    text(0.05, 0.85, sprintf('integral 1 = %5.3f',sum(real(d1{1}))), 'units', 'normalized')
  end
catch err
end

% -------------------------------------------------------------------------
function cbAx_Callback(hObject, eventdata, handles)
hh = [handles.cbAx1, handles.cbAx2, handles.cbAx3, handles.cbAx4];
set(hh(hh~=hObject),'value',false)
hh1 = [handles.sl1, handles.sl2, handles.sl3, handles.sl4];
set(hh1(hh==hObject),'Visible','off')
set(hh1(hh~=hObject),'Visible','on');
PlotIt(handles)
