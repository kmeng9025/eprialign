function varargout = AdditionGUI(varargin)
% ADDITIONGUI MATLAB code for AdditionGUI.fig
%      ADDITIONGUI, by itself, creates a new ADDITIONGUI or raises the existing
%      singleton*.
%
%      H = ADDITIONGUI returns the handle to a new ADDITIONGUI or the handle to
%      the existing singleton*.
%
%      ADDITIONGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ADDITIONGUI.M with the given input arguments.
%
%      ADDITIONGUI('Property','Value',...) creates a new ADDITIONGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before AdditionGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to AdditionGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help AdditionGUI

% Last Modified by GUIDE v2.5 08-Dec-2015 15:38:10

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @AdditionGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @AdditionGUI_OutputFcn, ...
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


% --- Executes just before AdditionGUI is made visible.
function AdditionGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to AdditionGUI (see VARARGIN)

% Choose default command line output for AdditionGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes AdditionGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = AdditionGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in Calculate.
function Calculate_Callback(hObject, eventdata, handles)
% hObject    handle to Calculate (see GCBO)
 % handles = guidata(figure1);
  X_final =  str2double(handles.X_CT.String{1}) + str2double(handles.X_rad_adj.String{1})
  set(handles.X_final_position,'String',num2str(X_final))
  Y_final =  str2double(handles.Y_CT.String{1}) + str2double(handles.Y_rad_adj.String{1})
  set(handles.Y_final_position,'String',num2str(Y_final))
  Z_final =  str2double(handles.Z_CT.String{1}) + str2double(handles.Z_rad_adj.String{1})
  set(handles.Z_final_position,'String',num2str(Z_final))
  
  
  
  
  
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function X_CT_Callback(hObject, eventdata, handles)
% hObject    handle to X_CT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of X_CT as text
%        str2double(get(hObject,'String')) returns contents of X_CT as a double


% --- Executes during object creation, after setting all properties.
function X_CT_CreateFcn(hObject, eventdata, handles)
% hObject    handle to X_CT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function X_rad_adj_Callback(hObject, eventdata, handles)
% hObject    handle to X_rad_adj (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of X_rad_adj as text
%        str2double(get(hObject,'String')) returns contents of X_rad_adj as a double


% --- Executes during object creation, after setting all properties.
function X_rad_adj_CreateFcn(hObject, eventdata, handles)
% hObject    handle to X_rad_adj (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function X_final_position_Callback(hObject, eventdata, handles)
% hObject    handle to X_final_position (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of X_final_position as text
%        str2double(get(hObject,'String')) returns contents of X_final_position as a double


% --- Executes during object creation, after setting all properties.
function X_final_position_CreateFcn(hObject, eventdata, handles)
% hObject    handle to X_final_position (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Y_CT_Callback(hObject, eventdata, handles)
% hObject    handle to Y_CT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Y_CT as text
%        str2double(get(hObject,'String')) returns contents of Y_CT as a double


% --- Executes during object creation, after setting all properties.
function Y_CT_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Y_CT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Y_rad_adj_Callback(hObject, eventdata, handles)
% hObject    handle to Y_rad_adj (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Y_rad_adj as text
%        str2double(get(hObject,'String')) returns contents of Y_rad_adj as a double


% --- Executes during object creation, after setting all properties.
function Y_rad_adj_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Y_rad_adj (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Y_final_position_Callback(hObject, eventdata, handles)
% hObject    handle to Y_final_position (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Y_final_position as text
%        str2double(get(hObject,'String')) returns contents of Y_final_position as a double


% --- Executes during object creation, after setting all properties.
function Y_final_position_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Y_final_position (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Z_CT_Callback(hObject, eventdata, handles)
% hObject    handle to Z_CT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Z_CT as text
%        str2double(get(hObject,'String')) returns contents of Z_CT as a double


% --- Executes during object creation, after setting all properties.
function Z_CT_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Z_CT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Z_rad_adj_Callback(hObject, eventdata, handles)
% hObject    handle to Z_rad_adj (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Z_rad_adj as text
%        str2double(get(hObject,'String')) returns contents of Z_rad_adj as a double


% --- Executes during object creation, after setting all properties.
function Z_rad_adj_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Z_rad_adj (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit9_Callback(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit9 as text
%        str2double(get(hObject,'String')) returns contents of edit9 as a double


% --- Executes during object creation, after setting all properties.
function edit9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
