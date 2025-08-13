function varargout = SliceGalleryRPT(varargin)
% SLICEGALLERYRPT MATLAB code for SliceGalleryRPT.fig
%      SLICEGALLERYRPT, by itself, creates a new SLICEGALLERYRPT or raises the existing
%      singleton*.
%
%      H = SLICEGALLERYRPT returns the handle to a new SLICEGALLERYRPT or the handle to
%      the existing singleton*.
%
%      SLICEGALLERYRPT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SLICEGALLERYRPT.M with the given input arguments.
%
%      SLICEGALLERYRPT('Property','Value',...) creates a new SLICEGALLERYRPT or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SliceGalleryRPT_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SliceGalleryRPT_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SliceGalleryRPT

% Last Modified by GUIDE v2.5 29-May-2014 12:32:04

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SliceGalleryRPT_OpeningFcn, ...
                   'gui_OutputFcn',  @SliceGalleryRPT_OutputFcn, ...
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


% --- Executes just before SliceGalleryRPT is made visible.
function SliceGalleryRPT_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SliceGalleryRPT (see VARARGIN)

% Choose default command line output for SliceGalleryRPT
handles.output = hObject;
handles.hh = varargin{1};

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SliceGalleryRPT wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --------------------------------------------------------------------
function varargout = SliceGalleryRPT_OutputFcn(hObject, eventdata, handles) 

% Get default command line output from handles structure
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pushbutton1_Callback(hObject, eventdata, handles)
hfig = str2double(get(handles.edit1, 'string'));
figure(hfig);

% Load complete list of 2D files
plist = {'FullName', 'data'};
find_list2D    = arbuz_FindImage(handles.hh, 'v', 'ImageType', '2D', plist);
nslices = length(find_list2D);

for ii=1:length(find_list2D)
  subplot(1, nslices, ii);
  sub_pos = get(gca,'position'); % get subplot axis position
  set(gca,'position',sub_pos.*[1 1 1.25 1.25]) % stretch its width and height
  image(find_list2D{ii}.data);
  axis('image'); axis('off');
  title(find_list2D{ii}.FullName, 'Interpreter', 'none')
end

tightfig;

% --------------------------------------------------------------------
function pushbutton2_Callback(hObject, eventdata, handles)
hfig = str2double(get(handles.edit1, 'string'));
figure(hfig);
