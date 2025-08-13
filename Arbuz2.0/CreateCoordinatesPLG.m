function varargout = CreateCoordinatesPLG(varargin)
% CREATECOORDINATESPLG M-file for CreateCoordinatesPLG.fig
% Co-Registration GUI and plug-ins 

% University of Chicago Medical Center
% Department of Radiation and Cellular Oncology
% B. Epel    2008

% Last Modified by GUIDE v2.5 16-Jun-2008 15:22:20

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CreateCoordinatesPLG_OpeningFcn, ...
                   'gui_OutputFcn',  @CreateCoordinatesPLG_OutputFcn, ...
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
function CreateCoordinatesPLG_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to CreateCoordinatesPLG (see VARARGIN)

% Choose default command line output for CreateCoordinatesPLG
handles.output = hObject;
handles.hh = varargin{1};

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes CreateCoordinatesPLG wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --------------------------------------------------------------------
function varargout = CreateCoordinatesPLG_OutputFcn(hObject, eventdata, handles) 
% Get default command line output from handles structure
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pbVectorDone_Callback(hObject, eventdata, handles)

origin = str2num(get(handles.eVectorOrigin, 'String'));
z      = str2num(get(handles.eVectorVector, 'String'));

x = [0 z(3) -z(2)]; x=x/norm(x);
y = cross(z,x);

rotation = [x,0;y,0;z,0;0,0,0,1]';

coord.Name = 'CC1';
coord.A    = inv(rotation) * inv(hmatrix_translate(-origin));
arbuz_set(handles.hh, 'Coordinates', {coord});

arbuz_UpdateInterface(handles.hh);

% --------------------------------------------------------------------
function pbVectorBuildFromAnchors_Callback(hObject, eventdata, handles)
Anchors = arbuz_FindImage(handles.hh, 'all', 'Name', get(handles.edit1, 'String'), {'data','A','Apre','Aproxy'});

for ii=1:length(Anchors)
  r = htransform_vectors(Anchors{ii}.Apre*Anchors{ii}.A, Anchors{ii}.data);
  fiducial(ii,:) = r(1:3);
end

[origin,z]=fit_line_through_points(fiducial);
set(handles.eVectorOrigin, 'String', num2str(origin));
set(handles.eVectorVector, 'String', num2str(z));

% --------------------------------------------------------------------
function pbUpdateOriginAnchor_Callback(hObject, eventdata, handles)
Anchors = arbuz_FindImage(handles.hh, 'proxy', 'ImageType', 'XYZ', {'FullName'});

str = {};
for ii=1:length(Anchors), str{end+1} = Anchors{ii}.FullName; end
set(handles.pmOriginAnchorList, 'String', str);

% --------------------------------------------------------------------
function pbDoneOriginAnchor_Callback(hObject, eventdata, handles)
str = get(handles.pmOriginAnchorList, 'String');
val = get(handles.pmOriginAnchorList, 'Value');

Anchors = arbuz_FindImage(handles.hh, 'proxy', 'FullName', str{val}, {'FullName', 'data', 'Acurrent'});

if ~isempty(Anchors)
  name = inputdlg('Coordinates name','Input', 1, {'Coord1'});

  if ~isempty(name)
    hhandles = guidata(handles.hh);

    coord.Name = name{1};
    XYZ = htransform_vectors(Anchors{1}.Acurrent, Anchors{1}.data);
    coord.A    = hmatrix_translate(XYZ);
    hhandles.Coordinates{1} = coord;
    guidata(hhandles.MainFigure,hhandles);
  end
end

arbuz_UpdateInterface(hhandles);

%--------------------------------------------------------------------------
function pbVectorGetFromFigure_Callback(hObject, eventdata, handles)

hh = str2double(get(handles.eGetFromFigure, 'String'));

if ishandle(hh)
  figure(hh);
  ginput(1);
  pp = get(gca, 'CurrentPoint');
  tt = pp(1, :) - pp(2, :);
  
  % choose an origin, a point on the line closest to (0,0,0)
  u = sum(- pp(1,:).*tt)/norm(tt)^2;
  orgn = pp(1,:) + u*tt;
  tt = tt/norm(tt);

  set(handles.eVectorOrigin, 'String', num2str(orgn));
  set(handles.eVectorVector, 'String', num2str(tt));
  
  pbVectorDone_Callback(hObject, eventdata, handles);
  arbuz_RedrawAll(handles.hh)
end

if get(handles.cbClose, 'value')
  closereq;
end
