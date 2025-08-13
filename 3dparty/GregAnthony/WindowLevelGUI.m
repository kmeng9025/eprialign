%Thresholding GUI
%Greg Anthony
%2015

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function varargout = WindowLevelGUI(varargin)
%WINDOWLEVELGUI M-file for WindowLevelGUI.fig
%      WINDOWLEVELGUI, by itself, creates a new WINDOWLEVELGUI or raises the existing
%      singleton*.
%
%      H = WINDOWLEVELGUI returns the handle to a new WINDOWLEVELGUI or the handle to
%      the existing singleton*.
%
%      WINDOWLEVELGUI('Property','Value',...) creates a new WINDOWLEVELGUI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to WindowLevelGUI_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      WINDOWLEVELGUI('CALLBACK') and WINDOWLEVELGUI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in WINDOWLEVELGUI.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help WindowLevelGUI

% Last Modified by GUIDE v2.5 05-Jun-2015 12:06:28

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @WindowLevelGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @WindowLevelGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
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


% --- Executes just before WindowLevelGUI is made visible.
function WindowLevelGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to WindowLevelGUI (see VARARGIN)

% Choose default command line output for WindowLevelGUI

if(nargin == 7)
    handles.Image = varargin{1};
    handles.Start = varargin{2};
    handles.End = varargin{3};
    handles.Orig = varargin{4};
end

% Initialize handles structure
handles.ImEdit = handles.Image;
handles.direc = 1;
handles.mode = 1;
handles.slice = handles.Start;
handles.LoThresh = 0;
handles.HiThresh = max(handles.Image(:));
handles.saveTog = handles.ImEdit;

% initialize GUI controls
set(handles.SL_slice, 'Min', handles.Start, 'Max', handles.End, 'Value', handles.Start, 'SliderStep', [1/(handles.End-handles.Start),1/(handles.End-handles.Start)])
set(handles.SL_Low, 'Min', 0, 'Max', max(handles.Image(:)), 'Value', handles.LoThresh, 'SliderStep', [1/max(handles.Image(:)),100/max(handles.Image(:))])
set(handles.SL_High, 'Min', 0, 'Max', max(handles.Image(:)), 'Value', handles.HiThresh, 'SliderStep', [1/max(handles.Image(:)),100/max(handles.Image(:))])
set(handles.txt_slice, 'String', {'Slice:',num2str(handles.slice)},'Max',2)
set(handles.txt_Low, 'String', {'Lower Threshold:',num2str(handles.LoThresh)},'Max',2)
set(handles.txt_high, 'String', {'Upper Threshold:',num2str(handles.HiThresh)},'Max',2)
set(handles.Popup_direction, 'Value', 1)
set(handles.togglebutton3, 'Value', 0)

axes(handles.axes1)
imshow(handles.ImEdit(:,:,handles.slice),[0,max(handles.Image(:))])
colormap bone

% Choose default command line output for SetCoordinateSystem
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SetCoordinateSystem wait for user response (see UIRESUME)
uiwait(handles.figure1)


% --- Outputs from this function are returned to the command line.
function varargout = WindowLevelGUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

varargout{1} = handles.output;
delete(handles.figure1);

%--------------------------------------------------------------------------

%Visualization Function
function vis_update(hObject, handles)
axes(handles.axes1)  

view = get(handles.togglebutton3,'Value');
if view == 0
    imshow(handles.ImEdit(:,:,handles.slice),[0,max(handles.Image(:))])
    colormap bone
elseif view == 1
    imshow(handles.Image(:,:,handles.slice),[0,max(handles.Image(:))])
    colormap bone
end
    
%--------------------------------------------------------------------------

% --- Executes on slider movement.
function SL_slice_Callback(hObject, eventdata, handles)
% hObject    handle to SL_slice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

handles.slice = get(handles.SL_slice, 'Value');
slice = handles.ImEdit(:,:,handles.slice);
PixOn = slice(slice~=0);

set(handles.SL_Low,'Value',min(PixOn));
set(handles.SL_High,'Value',max(PixOn));
txt_Low_Callback(hObject, eventdata, handles)
txt_high_Callback(hObject, eventdata, handles)

guidata(hObject, handles);
vis_update(hObject, handles);
txt_slice_Callback(hObject, eventdata, handles)



% --- Executes during object creation, after setting all properties.
function SL_slice_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SL_slice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


function txt_slice_Callback(hObject, eventdata, handles)
% hObject    handle to txt_slice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.slice = get(handles.SL_slice,'Value');
set(handles.txt_slice, 'String', {'Slice:',num2str(handles.slice)})

guidata(hObject, handles)

% Hints: get(hObject,'String') returns contents of txt_slice as text
%        str2double(get(hObject,'String')) returns contents of txt_slice as a double


% --- Executes during object creation, after setting all properties.
function txt_slice_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_slice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes on selection change in Popup_direction.
function Popup_direction_Callback(hObject, eventdata, handles)
% hObject    handle to Popup_direction (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns Popup_direction contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Popup_direction

handles.direc = get(handles.Popup_direction,'Value');
handles.mode = handles.direc;

guidata(hObject, handles)
vis_update(hObject, handles)

% --- Executes during object creation, after setting all properties.
function Popup_direction_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Popup_direction (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function txt_Low_Callback(hObject, eventdata, handles)
% hObject    handle to txt_Low (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txt_Low as text
%        str2double(get(hObject,'String')) returns contents of txt_Low as a double

handles.LoThresh = get(handles.SL_Low,'Value');
set(handles.txt_Low, 'String', {'Lower Threshold:',num2str(handles.LoThresh)})

guidata(hObject, handles)

% --- Executes during object creation, after setting all properties.
function txt_Low_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_Low (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes on slider movement.
function SL_Low_Callback(hObject, eventdata, handles)
% hObject    handle to SL_Low (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

%Set threshold and edit image

if handles.mode == 1
    warndlg('Manual threshold adjustment can only be performed in single slice editing mode','Mode Change Required','modal');
    
elseif handles.mode == 2
    handles.HiThresh = get(handles.SL_High,'Value');
    handles.LoThresh = get(handles.SL_Low,'Value');
    slice = handles.Image(:,:,handles.slice);
    slice(slice < handles.LoThresh)=0;
    slice(slice > handles.HiThresh)=0;
    handles.ImEdit(:,:,handles.slice) = slice;
end

guidata(hObject,handles)
vis_update(hObject, handles)
txt_Low_Callback(hObject, eventdata, handles)



% --- Executes during object creation, after setting all properties.
function SL_Low_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SL_Low (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



% --- Executes on slider movement.
function SL_High_Callback(hObject, eventdata, handles)
% hObject    handle to SL_High (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

%Set threshold and edit image

if handles.mode == 1
    warndlg('Manual threshold adjustment can only be performed in single slice editing mode','Mode Change Required','modal');
    
elseif handles.mode == 2
    handles.HiThresh = get(handles.SL_High,'Value');
    handles.LoThresh = get(handles.SL_Low,'Value');
    slice = handles.Image(:,:,handles.slice);
    slice(slice < handles.LoThresh)=0;
    slice(slice > handles.HiThresh)=0;
    handles.ImEdit(:,:,handles.slice) = slice;
end

guidata(hObject,handles)
vis_update(hObject, handles)
txt_high_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function SL_High_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SL_High (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function txt_high_Callback(hObject, eventdata, handles)
% hObject    handle to txt_high (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txt_high as text
%        str2double(get(hObject,'String')) returns contents of txt_high as a double

handles.HiThresh = get(handles.SL_High,'Value');
set(handles.txt_high, 'String', {'Upper Threshold:',num2str(handles.HiThresh)})

guidata(hObject, handles)


% --- Executes during object creation, after setting all properties.
function txt_high_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_high (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes on button press in PB_Otsu.
function PB_Otsu_Callback(hObject, eventdata, handles)
% hObject    handle to PB_Otsu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if handles.mode == 2
    warndlg('Otsu thresholding can only be performed in global editing mode','Mode Change Required','modal');
    
else

prompt = {'Enter number of automatic thresholds to locate (2 recommended)'};
numThresh = inputdlg(prompt, 'Otsu Thresholding', 1, {'2'});

for i=handles.Start:handles.End
    NormImage = (1/max(handles.Image(:)))*handles.Image;
    Levels = multithresh(NormImage(NormImage~=0),str2double(numThresh));
    handles.ImEdit(:,:,i) = imquantize(NormImage(:,:,i),[0.0001,Levels])-1;
end

guidata(hObject,handles)
axes(handles.axes1)
imshow(handles.ImEdit(:,:,handles.slice),[0,str2double(numThresh)+1])
colormap default

prompt = {'Enter color label of region most closely matching tumor volume'};
Label = inputdlg(prompt, 'Otsu Thresholding', 1, {'2'});

for i=1:size(handles.Image,3)
    MaskSlice = handles.ImEdit(:,:,i);
    MaskSlice(MaskSlice~=str2double(Label))=0;
    MaskSlice(MaskSlice==str2double(Label))=1;
    handles.ImEdit(:,:,i) = MaskSlice.*handles.Image(:,:,i);
end

if str2double(Label)==1
    handles.LoThresh = min(handles.Image(handles.Image~=0));
    handles.HiThresh = (max(handles.Image(:)))*Levels(str2double(Label));
elseif str2double(Label)==str2double(numThresh)+1
    handles.LoThresh = (max(handles.Image(:)))*Levels(str2double(Label)-1);
    handles.HiThresh = max(handles.Image(handles.Image~=0));
else
    handles.LoThresh = (max(handles.Image(:)))*Levels(str2double(Label)-1);
    handles.HiThresh = (max(handles.Image(:)))*Levels(str2double(Label));
    set(handles.SL_Low, 'Value', handles.LoThresh)
    set(handles.SL_High, 'Value', handles.HiThresh)
end

guidata(hObject,handles)
txt_Low_Callback(hObject, eventdata, handles)
txt_high_Callback(hObject, eventdata, handles)

vis_update(hObject, handles)

end


% --- Executes on button press in PB_Reset.
function PB_Reset_Callback(hObject, eventdata, handles)
% hObject    handle to PB_Reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.ImEdit = handles.Image;
handles.LoThresh = 0;
handles.HiThresh = max(handles.Image(:));
set(handles.SL_Low, 'Value', handles.LoThresh)
set(handles.SL_High, 'Value', handles.HiThresh)

guidata(hObject,handles)
txt_Low_Callback(hObject, eventdata, handles)
txt_high_Callback(hObject, eventdata, handles)

vis_update(hObject, handles)


% --- Executes on button press in PB_Confirm.
function PB_Confirm_Callback(hObject, eventdata, handles)
% hObject    handle to PB_Confirm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

prompt = {'Is this the thresholded mask you want to start from?'};
choice = questdlg(prompt, 'Confirm Output', 'Yes', 'No', 'No');

switch choice
    case 'Yes'
        handles.output = handles.ImEdit;
        guidata(hObject, handles);
        uiresume(handles.figure1);
    case 'No'
end


% --- Executes on button press in togglebutton3.
function togglebutton3_Callback(hObject, eventdata, handles)
% hObject    handle to togglebutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of togglebutton3

vis_update(hObject, handles)

%%%
