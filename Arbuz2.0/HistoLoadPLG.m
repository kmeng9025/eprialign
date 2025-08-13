function varargout = HistoLoadPLG(varargin)
% HISTOLOADPLG MATLAB code for HistoLoadPLG.fig
%      HISTOLOADPLG, by itself, creates a new HISTOLOADPLG or raises the existing
%      singleton*.
%
%      H = HISTOLOADPLG returns the handle to a new HISTOLOADPLG or the handle to
%      the existing singleton*.
%
%      HISTOLOADPLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in HISTOLOADPLG.M with the given input arguments.
%
%      HISTOLOADPLG('Property','Value',...) creates a new HISTOLOADPLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before HistoLoadPLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to HistoLoadPLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help HistoLoadPLG

% Last Modified by GUIDE v2.5 06-May-2014 10:03:03

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @HistoLoadPLG_OpeningFcn, ...
                   'gui_OutputFcn',  @HistoLoadPLG_OutputFcn, ...
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


% --- Executes just before HistoLoadPLG is made visible.
function HistoLoadPLG_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to HistoLoadPLG (see VARARGIN)

% Choose default command line output for HistoLoadPLG
handles.output = hObject;
handles.hh = varargin{1};
handles.image_name = {'1H', '1T', '2H', '2T', '3H', '3T', '4H', '4T', '5H', '5T', '6H', '6T', '7H', '7T'};
pos = get(handles.figure1, 'Position');
the_top = pos(4)-4.5;
for ii=1:14
  handles.e{ii} = uicontrol('Style', 'edit', 'Parent', handles.figure1, 'Units', 'character', ...
    'Position', [1, the_top-2*ii, 63, 1.6], ...
    'BackgroundColor', [1,1,1], 'Tag', num2str(ii)); 
  handles.pb{ii} = uicontrol('Style', 'pushbutton', 'Parent', handles.figure1, 'Units', 'character', ...
    'Position', [64, the_top-2*ii, 8, 1.6], ...
    'String', handles.image_name{ii}, 'Tag', num2str(ii), 'Callback', 'HistoLoadPLG(''pbSlices_Callback'',gco,[],guidata(gco))');
end

handles.dir = pwd;

% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function varargout = HistoLoadPLG_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;


% --------------------------------------------------------------------
function pbLoadSlices_Callback(hObject, eventdata, handles)

for ii=1:14
  fname = get(handles.e{ii}, 'String');
  if ~isempty(fname)
    new_image.Name = handles.image_name{ii};
    new_image.ImageType = '2D';
    new_image.FileName = fname;
    [new_image.data, new_image.data_info] = arbuz_LoadImage(fname, new_image.ImageType);
    new_image.box = safeget(new_image.data_info, 'Bbox', size(new_image.data));
    new_image.isLoaded = 1;
    new_image.isStore = 1;
    arbuz_AddImage(handles.hh, new_image);
  end
end
arbuz_UpdateInterface(handles.hh);

% --------------------------------------------------------------------
function pbSlices_Callback(hObject, eventdata, handles)

ii = str2double(get(hObject, 'Tag'));
[filename, pathname] = uigetfile( ...
  {'*.jpg;*.jpeg;*.bmp;*.tiff;*.tif', 'Image Files (*.jpg, *.jpeg, *.bmp, *.tiff, *.tif)'; ...
  '*.*',                   'All Files (*.*)'}, ...
  'Pick a file', handles.dir);
if isequal(filename,0) || isequal(pathname,0)
else
  set(handles.e{ii}, 'String', fullfile(pathname, filename));
  handles.dir = pathname;
  guidata(hObject, handles);
end
