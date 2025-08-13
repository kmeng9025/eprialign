function varargout = ImageSetScalePLG(varargin)
% IMAGESETSCALEPLG M-file for ImageSetScalePLG.fig
% Co-Registration GUI and plug-ins 

% University of Chicago Medical Center
% Department of Radiation and Cellular Oncology
% B. Epel    2007

% Last Modified by GUIDE v2.5 15-Jun-2016 15:34:11

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ImageSetScalePLG_OpeningFcn, ...
                   'gui_OutputFcn',  @ImageSetScalePLG_OutputFcn, ...
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
function ImageSetScalePLG_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for ImageSetScalePLG
handles.output = hObject;
handles.hh = varargin{1};

% Update handles structure
guidata(hObject, handles);


% --------------------------------------------------------------------
function varargout = ImageSetScalePLG_OutputFcn(hObject, eventdata, handles) 

% Get default command line output from handles structure
varargout{1} = handles.output;


% --------------------------------------------------------------------
function pbShowOnScreenRuler_Callback(hObject, eventdata, handles)

if isempty(safeget(handles, 'ruler', []))
  handles.ruler      = imdistline(handles.axes1);
  handles.ruler_type = 'imdistline';
  guidata(handles.figure1, handles);
end


% --------------------------------------------------------------------
function pbShowOnScreenCircRuler_Callback(hObject, eventdata, handles)

if isempty(safeget(handles, 'ruler', []))
  handles.ruler      = imellipse(handles.axes1, []);
  handles.ruler_type = 'imellipse';

  % set aspect ratio to 1
  api = iptgetapi(handles.ruler);
  pos = api.getPosition(); pos(3:4) = mean(pos(3:4));
  api.setPosition(pos);
  api.setFixedAspectRatioMode(true);

  guidata(handles.figure1, handles);
end

% --------------------------------------------------------------------
function eImageXResolution_Callback(hObject, eventdata, handles)


% --------------------------------------------------------------------
function pbUpdateFigure_Callback(hObject, eventdata, handles)
find_list = GetSelectedImage(handles);

if ~isempty(find_list)
  im = arbuz_FindImage(handles.hh, {find_list}, '', '', {'data','Aslave','A','Name','FileName'});
  im = im{1};
  im.max = max(im.data(:));
  im.min = min(im.data(:));
else
  im = [];
end

handles.image = im;
handles.image_data = im.data;

if ndims(im) == 2
  handles.image.type = '2D';
else
  handles.image.type = 'RGB';
end
guidata(hObject, handles);

sc = hmatrix_scale_get(handles.image.A);
set(handles.eImageXResolution, 'String', num2str(sc(1)))
set(handles.eImageYResolution, 'String', num2str(sc(2)))

set(handles.figure1, 'Name', sprintf('%s [%s]',handles.image.Name, epr_ShortFileName(handles.image.FileName, 50)))
if isempty(handles.image.data)
  disp('Photo2DscalePLG: Data are empty.');
  return;
end
DrawImage(handles);

% --------------------------------------------------------------------
function eImageYResolution_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function DrawImage(handles)
val1 = get(handles.slAdj1, 'Value');
val2 = get(handles.slAdj2, 'Value');
di = handles.image.max - handles.image.min;
v1 = handles.image.min + di * val1;
v2 = handles.image.min + di * val2;

% imagesc(handles.image_data,'Parent',handles.axes1)
switch handles.image.type
  case 'RGB'
    image(imadjust(handles.image_data, [v1, v1, v1; v2, v2, v2]),'Parent',handles.axes1)
  case '2D'  
    axes(handles.axes1);
    imagesc(handles.image_data, [v1, v2]);
end
axis(handles.axes1, 'image')


% --------------------------------------------------------------------
function pbReadRulerData_Callback(hObject, eventdata, handles)

if ~isempty(safeget(handles, 'ruler', []))
    try api = iptgetapi(handles.ruler); 
    catch
      handles.imdistline = []; guidata(handles.figure1, handles); return;
    end
    default_answer = num2str(safeget(handles, 'ruler_length_mm', 20));
    switch handles.ruler_type
      case 'imdistline'
        pos = api.getPosition();
        handles.ruler_length_pixels = api.getDistance();
        handles.ruler_center = [pos(1), pos(2)] + handles.ruler_length_pixels/2;
        handles.ruler_length_mm = str2double(inputdlg('The length of ruler, mm', 'Input', 1, {default_answer}));
      case 'imellipse'
        pos = api.getPosition();
        handles.ruler_length_pixels = mean(pos(3:4));
        handles.ruler_center = [pos(1), pos(2)] + handles.ruler_length_pixels/2;
        handles.ruler_length_mm = str2double(inputdlg('The diameter of the circle, mm', 'Input', 1, {default_answer}));
    end
    
    api.delete();
    handles.ruler = [];

    guidata(handles.figure1, handles);
    
    set(handles.eImageXResolution, 'String', num2str(handles.ruler_length_mm/handles.ruler_length_pixels))
    set(handles.eImageYResolution, 'String', num2str(handles.ruler_length_mm/handles.ruler_length_pixels))
end

% --------------------------------------------------------------------
function pbDone_Callback(hObject, eventdata, handles)

%H = hmatrix_translate(-[handles.ruler_center(1), handles.ruler_center(2), 0]) * hmatrix_scale(handles.ruler_length_mm/handles.ruler_length_pixels*[1,1,1]);

H = hmatrix_translate(-[str2num(handles.eImageXResolution.String), str2num(handles.eImageYResolution.String), 0]) * hmatrix_scale((str2num(handles.eImageYResolution.String)*[1,1,1]));

arbuz_SetImage(handles.hh, {handles.image}, 'Anative', H);
arbuz_SetTransformation(handles.hh, 'T1', handles.image.Image,H);
arbuz_ShowMessage(handles.hh, sprintf('Native transformation is assigned to %s image.', handles.image.Image));

arbuz_SetActiveTransformation(handles.hh);

% --------------------------------------------------------------------
function figure1_ResizeFcn(hObject, eventdata, handles)
fig_size = get(handles.figure1, 'Position');

tool_size = 28;
wborder = 1.5;
hborder = 0.5;
w_axborder = 6;
h_axborder = 2;
h_buttonsize = 2.2;
h_buttonsize_tools = 1.8;

h_buttonsize_update = 2.6;
w_buttonsize_update = 6.6;

panel_size = [fig_size(3)-tool_size-wborder, hborder, tool_size, fig_size(4)-2*hborder];
set(handles.panelTools, 'Position',panel_size);

set(handles.axes1, 'Position', ...
  [wborder+w_axborder, hborder+h_axborder, ...
  fig_size(3)-tool_size-3*wborder-w_axborder, ...
  fig_size(4)-2*hborder-h_axborder]);

set(handles.pbUpdateFigure, 'Position', ...
  [wborder, ...
  panel_size(4) - 2*hborder - h_buttonsize,...
  panel_size(3)-2*wborder-w_buttonsize_update, h_buttonsize_update]);

set(handles.pbInfo, 'Position', ...
  [panel_size(3)-wborder-w_buttonsize_update, panel_size(4) - h_buttonsize_update - hborder-0.1, ...
  w_buttonsize_update, h_buttonsize_update]);

hhs = [handles.pbShowOnScreenRuler, handles.pbShowOnScreenCircRuler, ...
  handles.pbReadRulerData];
for ii=1:length(hhs)
set(hhs(ii), 'Position', ...
  [wborder, panel_size(4) - 4 - ii*h_buttonsize_tools-ii*hborder/2,...
  panel_size(3)-2*wborder, h_buttonsize_tools]);
end

hhs = [handles.tImageXResolution, handles.eImageXResolution, ...
  handles.tImageYResolution, handles.eImageYResolution];
for ii=1:length(hhs)
set(hhs(ii), 'Position', ...
  [wborder, panel_size(4) - 12 - ii*h_buttonsize_tools-ii*hborder/2,...
  panel_size(3)-2*wborder, h_buttonsize_tools]);
end

set(handles.pbDone, 'Position', ...
  [wborder, hborder, panel_size(3)-2*wborder, h_buttonsize]);

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
function slAdj1_Callback(hObject, eventdata, handles)
DrawImage(handles);

% --------------------------------------------------------------------
function slAdj2_Callback(hObject, eventdata, handles)
DrawImage(handles);

% --------------------------------------------------------------------
function pbInfo_Callback(hObject, eventdata, handles)
the_message = ['This 2D plugin assigns transformation from image pixels to mm.\n\n',...
  '1. Select image in ArbuzGUI. Press [Load Image] button.\n',...
  '2. Place a ruler on the photograph using [Show Ruler] or [Show Cricular Ruler] buttons.\n',...
  '3. Press [Read Ruler Data] button and give dimentsions of the ruler in mm.\n',...
  '4. Press [Apply] button to assign Native and T1 transformations of the object.'];
msgbox(sprintf(the_message),'Info', 'help')
