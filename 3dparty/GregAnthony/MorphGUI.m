%Thresholding GUI
%Greg Anthony
%2015

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function varargout = MorphGUI(varargin)
%MORPHGUI M-file for MorphGUI.fig
%      MORPHGUI, by itself, creates a new MORPHGUI or raises the existing
%      singleton*.
%
%      H = MORPHGUI returns the handle to a new MORPHGUI or the handle to
%      the existing singleton*.
%
%      MORPHGUI('Property','Value',...) creates a new MORPHGUI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to MorphGUI_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      MORPHGUI('CALLBACK') and MORPHGUI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in MORPHGUI.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MorphGUI

% Last Modified by GUIDE v2.5 01-Jun-2015 23:22:32

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MorphGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @MorphGUI_OutputFcn, ...
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


% --- Executes just before MorphGUI is made visible.
function MorphGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   unrecognized PropertyName/PropertyValue pairs from the
%            command line (see VARARGIN)

% Choose default command line output for MorphGUI

if(nargin == 7)
    handles.Image = varargin{1};
    handles.Start = varargin{2};
    handles.End = varargin{3};
    handles.Orig = varargin{4};
end

% Update handles structure
handles.ImEdit = handles.Image;
handles.Mask = handles.ImEdit;
handles.Mask(handles.Mask~=0)=1;
handles.slice = handles.Start;
handles.mode = 1;
handles.strel = 1;
handles.strelsize = 1;
handles.shapes = {'ball', 'disk', 'diamond', 'square'};

handles.prev = handles.ImEdit;
handles.saveTog = handles.ImEdit;

handles.output = handles.ImEdit;

% initialize GUI controls
set(handles.Pop_Mode, 'Value', 1)
set(handles.Pop_Strel, 'Value', 1)
set(handles.SL_Slice, 'Min', handles.Start, 'Max', handles.End, 'Value', handles.Start, 'SliderStep', [1/(handles.End-handles.Start),1/(handles.End-handles.Start)])
set(handles.SL_Strel, 'Min', 1, 'Max', 10, 'Value', handles.strelsize, 'SliderStep', [1/9,1/9])
set(handles.txt_slice, 'String', {num2str(handles.slice)})
set(handles.txt_strel, 'String', {'Size (pixels):',num2str(handles.strelsize)},'Max',2)
set(handles.togglebutton1, 'Value', 0)

axes(handles.axes1)
imshow(handles.ImEdit(:,:,handles.slice),[0,max(handles.Orig(:))])
colormap bone

% Choose default command line output for SetCoordinateSystem
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SetCoordinateSystem wait for user response (see UIRESUME)
uiwait(handles.figure1)


% --- Outputs from this function are returned to the command line.
function varargout = MorphGUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
delete(handles.figure1);

%--------------------------------------------------------------------------

%Visualization Function
function vis_update(hObject,handles)
axes(handles.axes1)

imshow(handles.ImEdit(:,:,handles.slice),[0,max(handles.Orig(:))])
colormap bone
    
%--------------------------------------------------------------------------
        

% --- Executes on slider movement.
function SL_Slice_Callback(hObject, eventdata, handles)
% hObject    handle to SL_Slice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

handles.slice = get(handles.SL_Slice, 'Value');

guidata(hObject, handles);
vis_update(hObject,handles);
txt_slice_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function SL_Slice_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SL_Slice (see GCBO)
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

% Hints: get(hObject,'String') returns contents of txt_slice as text
%        str2double(get(hObject,'String')) returns contents of txt_slice as a double

handles.slice = get(handles.SL_Slice,'Value');
set(handles.txt_slice, 'String', {num2str(handles.slice)})

guidata(hObject, handles)


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


% --- Executes on button press in PB_Hole_Fill.
function PB_Hole_Fill_Callback(hObject, eventdata, handles)
% hObject    handle to PB_Hole_Fill (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.prev = handles.ImEdit;
handles.Mask = zeros(size(handles.ImEdit));
handles.Mask(handles.ImEdit~=0)=1;

if handles.mode == 1
    handles.Mask(:,:,handles.slice) = imfill(handles.Mask(:,:,handles.slice),'holes');
    
elseif handles.mode == 2
    for i=1:size(handles.ImEdit,3)
        handles.Mask(:,:,i) = imfill(handles.Mask(:,:,i),'holes');
    end
end

handles.ImEdit = (handles.Mask).*(handles.Orig);

guidata(hObject,handles)
vis_update(hObject,handles)

% --- Executes on button press in PB_Remove_Specks.
function PB_Remove_Specks_Callback(hObject, eventdata, handles)
% hObject    handle to PB_Remove_Specks (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.prev = handles.ImEdit;
handles.Mask = zeros(size(handles.ImEdit));
handles.Mask(handles.ImEdit~=0)=1;

if handles.mode == 1
    %Label image regions
    [MaskLabel,numObjects(1,1)] = bwlabel(handles.Mask(:,:,handles.slice)); 
    Stats{1} = regionprops(MaskLabel);

    Slice = MaskLabel;
    %Keep only the largest region
    if numObjects(1,1)>1
        big = 1;
        for n=2:numObjects(1,1)
            if Stats{1}(n).Area > Stats{1}(big).Area
                Slice(Slice==big)=0;
                Slice(Slice==n)=1;
                big = n;
            else
                Slice(Slice==n)=0;
                Slice(Slice==big)=1;
            end
        end
    end
    handles.Mask(:,:,handles.slice) = Slice;
    
elseif handles.mode == 2
    %Label image regions
    for i=1:size(handles.Image,3)
        [MaskLabel(:,:,i),numObjects(i,1)] = bwlabel(handles.Mask(:,:,i)); 
        Stats{i} = regionprops(MaskLabel(:,:,i));
    end

    for i=1:size(handles.Image,3)
        Slice = MaskLabel(:,:,i);
    %Keep only the largest region
        if numObjects(i,1)>1
            big = 1;
            for n=2:numObjects(i,1)
                if Stats{i}(n).Area > Stats{i}(big).Area
                    Slice(Slice==big)=0;
                    Slice(Slice==n)=1;
                    big = n;
                else
                    Slice(Slice==n)=0;
                    Slice(Slice==big)=1;
                end
            end
        end
    handles.Mask(:,:,i) = Slice;
    end
end

handles.ImEdit = (handles.Mask).*(handles.Orig);

guidata(hObject,handles)
vis_update(hObject,handles)

% --- Executes on selection change in Pop_Mode.
function Pop_Mode_Callback(hObject, eventdata, handles)
% hObject    handle to Pop_Mode (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns Pop_Mode contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Pop_Mode

handles.mode = get(handles.Pop_Mode, 'Value');

guidata(hObject,handles)


% --- Executes during object creation, after setting all properties.
function Pop_Mode_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Pop_Mode (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in PB_Undo_Slice.
function PB_Undo_Slice_Callback(hObject, eventdata, handles)
% hObject    handle to PB_Undo_Slice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if handles.mode == 1
    handles.ImEdit(:,:,handles.slice) = handles.Image(:,:,handles.slice);
elseif handles.mode == 2
    warndlg('Switch to slice-by-slice edit mode to undo individual slices','Mode Change Required','modal');
end

guidata(hObject,handles)
vis_update(hObject,handles)

% --- Executes on button press in PB_Reset.
function PB_Reset_Callback(hObject, eventdata, handles)
% hObject    handle to PB_Reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.prev = handles.ImEdit;

prompt = {'Undo all morphological editing and start again from thresholded image?'};
choice = questdlg(prompt, 'Confirm Reset', 'Yes', 'No', 'No');

switch choice
    case 'Yes'
        handles.ImEdit = handles.Image;
        
        guidata(hObject,handles)
        vis_update(hObject,handles)
    case 'No'
end


% --- Executes on button press in PB_Confirm.
function PB_Confirm_Callback(hObject, eventdata, handles)
% hObject    handle to PB_Confirm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

prompt = {'Are you finished with all (non-manual) morphological editing?'};
choice = questdlg(prompt, 'Confirm Output', 'Yes', 'No', 'No');

switch choice
    case 'Yes'
        handles.output = handles.ImEdit;
        guidata(hObject, handles);
        uiresume(handles.figure1);
    case 'No'
end


% --- Executes on button press in PB_Undo.
function PB_Undo_Callback(hObject, eventdata, handles)
% hObject    handle to PB_Undo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.ImEdit = handles.prev;

guidata(hObject,handles)
vis_update(hObject,handles)


% --- Executes on button press in PB_Cancel.
function PB_Cancel_Callback(hObject, eventdata, handles)
% hObject    handle to PB_Cancel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

prompt = {'Delete this mask and return to thresholding GUI?'};
choice = questdlg(prompt, 'Cancel', 'Yes', 'No', 'No');

switch choice
    case 'Yes'
        handles.output = 0;
        guidata(hObject, handles);
        uiresume(handles.figure1);
    case 'No'
end


% --- Executes on selection change in Pop_Strel.
function Pop_Strel_Callback(hObject, eventdata, handles)
% hObject    handle to Pop_Strel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns Pop_Strel contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Pop_Strel

handles.strel = get(handles.Pop_Strel, 'Value');

guidata(hObject, handles)

   
% --- Executes during object creation, after setting all properties.
function Pop_Strel_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Pop_Strel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function SL_Strel_Callback(hObject, eventdata, handles)
% hObject    handle to SL_Strel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

handles.strelsize = round(get(handles.SL_Strel, 'Value'));

guidata(hObject, handles)
txt_strel_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function SL_Strel_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SL_Strel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function txt_strel_Callback(hObject, eventdata, handles)
% hObject    handle to txt_strel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txt_strel as text
%        str2double(get(hObject,'String')) returns contents of txt_strel as a double

set(handles.txt_strel, 'String', {'Size (pixels):',num2str(handles.strelsize)})

guidata(hObject, handles)


% --- Executes during object creation, after setting all properties.
function txt_strel_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_strel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in PB_Erode.
function PB_Erode_Callback(hObject, eventdata, handles)
% hObject    handle to PB_Erode (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if handles.strel == 1,
    SE = strel(handles.shapes{handles.strel},handles.strelsize,0,0);
else
    SE = strel(handles.shapes{handles.strel},handles.strelsize,0);
end

handles.prev = handles.ImEdit;
handles.Mask = zeros(size(handles.ImEdit));
handles.Mask(handles.ImEdit~=0)=1;

if handles.mode == 1
    handles.Mask(:,:,handles.slice) = imerode(handles.Mask(:,:,handles.slice),SE);
    
elseif handles.mode == 2
    for i=1:size(handles.Image,3)
        handles.Mask(:,:,i) = imerode(handles.Mask(:,:,i),SE);
    end
end

handles.ImEdit = (handles.Mask).*(handles.Orig);

guidata(hObject,handles)
vis_update(hObject,handles)


% --- Executes on button press in PB_Dilate.
function PB_Dilate_Callback(hObject, eventdata, handles)
% hObject    handle to PB_Dilate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if handles.strel == 1,
    SE = strel(handles.shapes{handles.strel},handles.strelsize,0,0);
else
    SE = strel(handles.shapes{handles.strel},handles.strelsize,0);
end

handles.prev = handles.ImEdit;
handles.Mask = zeros(size(handles.ImEdit));
handles.Mask(handles.ImEdit~=0)=1;

if handles.mode == 1
    handles.Mask(:,:,handles.slice) = imdilate(handles.Mask(:,:,handles.slice),SE);
    
elseif handles.mode == 2
    for i=1:size(handles.Image,3)
        handles.Mask(:,:,i) = imdilate(handles.Mask(:,:,i),SE);
    end
end

handles.ImEdit = (handles.Mask).*(handles.Orig);

guidata(hObject,handles)
vis_update(hObject,handles)


% --- Executes on button press in togglebutton1.
function togglebutton1_Callback(hObject, eventdata, handles)
% hObject    handle to togglebutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of togglebutton1

Tog = get(handles.togglebutton1, 'Value');
if Tog == 1
    handles.saveTog = handles.ImEdit;
    handles.ImEdit = handles.Orig;
    
    guidata(hObject,handles)
    vis_update(hObject,handles)
elseif Tog == 0
    handles.ImEdit = handles.saveTog;
    
    guidata(hObject,handles)
    vis_update(hObject,handles)
end


% --- Executes on button press in PB_Clear_Slice.
function PB_Clear_Slice_Callback(hObject, eventdata, handles)
% hObject    handle to PB_Clear_Slice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.ImEdit(:,:,handles.slice) = zeros(size(handles.Image,1),size(handles.Image,2));

guidata(hObject,handles)
vis_update(hObject,handles)

%%%