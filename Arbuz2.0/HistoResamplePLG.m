function varargout = HistoResamplePLG(varargin)
% HISTORESAMPLEPLG M-file for HistoResamplePLG.fig
% Co-Registration GUI and plug-ins 

% University of Chicago Medical Center
% Department of Radiation and Cellular Oncology
% B. Epel    2008

% Last Modified by GUIDE v2.5 04-Jun-2014 15:00:45

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @HistoResamplePLG_OpeningFcn, ...
                   'gui_OutputFcn',  @HistoResamplePLG_OutputFcn, ...
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


% --- Executes just before HistoResamplePLG is made visible.
function HistoResamplePLG_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to HistoResamplePLG (see VARARGIN)

% Choose default command line output for HistoResamplePLG
handles.output = hObject;
handles.hh = varargin{1};

handles.Rotate    = eye(4);
handles.Translate = eye(4);
handles.Scale     = eye(4);

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes HistoResamplePLG wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = HistoResamplePLG_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --------------------------------------------------------------------
function pbDone_Callback(hObject, eventdata, handles)

r = handles.image.Box;
corners = [1 1 0; r(2) 1 0; r(2) r(1) 0; 1 r(1) 0];

pix = str2double(get(handles.ePixels, 'String'));

AA = GetTransformationMatrix(handles);
tcorners=htransform_vectors(AA,corners);

sizeh = norm(tcorners(2,:)-tcorners(3,:));
sizew = norm(tcorners(1,:)-tcorners(2,:));
scale = pix/sizew;


[image_final] = resample_plane(handles.image.data, tcorners, pix, round(pix*sizeh/sizew));  

% add proxy image
default_name = [handles.image.Image, '_', num2str(pix)];
name = inputdlg('Resampled image name','Image Name', 1, {default_name});

if ~isempty(name)
  sz = size(image_final);
  MoveToCenter = hmatrix_translate(-[sz(2) sz(1) 0]/2);
  if get(handles.cbMirror, 'Value')
    Mirror   = hmatrix_rotate_y(180);
  else
    Mirror   = eye(4);
  end
  AAA = MoveToCenter*Mirror*hmatrix_scale([1/scale, 1/scale, 1])*....
    handles.Rotate*handles.Translate;
  proxy.data = image_final;
  proxy.A = AAA * handles.image.Aslave;
  proxy.Name = name{1};
  proxy.ImageType = '2D';
  proxy.isStore = 1;
  arbuz_AddImage(handles.hh, proxy, handles.image.ImageIdx);
  
  arbuz_UpdateInterface(handles.hh);
end

% --------------------------------------------------------------------
function pbLoad_Callback(hObject, eventdata, handles)
find_list = GetSelectedImage(handles);

if ~isempty(find_list)
  im = arbuz_FindImage(handles.hh, {find_list}, '', '', {'data','Name','FileName','Box', 'Aslave'});
  im = im{1};
  im.max = max(im.data(:));
  im.min = min(im.data(:));
else
  im = [];
end

handles.image = im;
handles.Translate = hmatrix_translate([handles.image.Box(2) handles.image.Box(1) 0]/2);
guidata(hObject, handles);

cla
image(handles.image.data);
axis image;  hold on

RedrawWireFrame(handles);

% --------------------------------------------------------------------
function slXYRotate_Callback(hObject, eventdata, handles)
pos = get(handles.slXYRotate, 'Value');
handles.Rotate = hmatrix_rotate_z(pos);
guidata(hObject, handles);
set(handles.eXYRotate, 'String', num2str(pos))

RedrawWireFrame(handles);

% --------------------------------------------------------------------
function slScale_Callback(hObject, eventdata, handles)
posx = get(handles.slXScale, 'Value');
posy = get(handles.slYScale, 'Value');
set(handles.eXScale, 'String', num2str(posx));
set(handles.eYScale, 'String', num2str(posy));

handles.Scale = hmatrix_scale([posx posy 1]);
guidata(hObject, handles);

RedrawWireFrame(handles);

% --------------------------------------------------------------------
function eXYRotate_Callback(hObject, eventdata, handles)


% --------------------------------------------------------------------
function eXScale_Callback(hObject, eventdata, handles)


% --------------------------------------------------------------------
function eYScale_Callback(hObject, eventdata, handles)


% --------------------------------------------------------------------
function cbMirror_Callback(hObject, eventdata, handles)

RedrawWireFrame(handles);


% --------------------------------------------------------------------
function pbImageCenter_Callback(hObject, eventdata, handles)

x = ginput(1);
handles.Translate = hmatrix_translate([x(1) x(2) 0]);
guidata(hObject, handles);

RedrawWireFrame(handles);


% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% ----- S E R V I C E     F U N C T I O N S --------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

function find_list = GetSelectedImage(handles)

find_list = arbuz_FindImage(handles.hh, 'all', 'Highlighted', 1, {});
find_list = arbuz_FindImage(handles.hh, find_list, 'ImageType', '2D', {});
find_list = find_list{1};

% --------------------------------------------------------------------
function AA = GetTransformationMatrix(handles)

r = handles.image.Box;
MoveToCenter = hmatrix_translate(-[r(2) r(1) 0]/2);
if get(handles.cbMirror, 'Value')
  Mirror   = hmatrix_rotate_y(180);
else
  Mirror   = eye(4);
end

AA = MoveToCenter*Mirror*handles.Scale*handles.Rotate*handles.Translate;

% --------------------------------------------------------------------
function RedrawWireFrame(handles)

obj = findobj(handles.axes1, 'Tag', 'line');
delete(obj);

r = handles.image.Box;

aa = min(r(1:2))/8;

corners = [   1   1  0 1;
  r(2)   1  0 1;
  r(2) r(1) 0 1;
  1    r(1) 0 1;
  1      1  0 1;
  r(2) r(1) 0 1;
  1    r(1) 0 1;
  r(2)   1 0 1;
  1      1  0 1;
  1  aa/8  0 1;
  r(2)  aa/8  0 1;
  r(2)  aa/4  0 1;
  1  aa/4  0 1;
  1  aa  0 1;
 aa   1  0 1  ];

AA = GetTransformationMatrix(handles);

for ii=1:size(corners,1), corners(ii,:) = corners(ii,:)*AA; end

line(corners(:,1), corners(:,2), 'Tag', 'line', 'Color','g', 'LineWidth', 2);
