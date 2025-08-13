function varargout = Whole_field_planPLG(varargin)
% WHOLE_FIELD_PLANPLG MATLAB code for Whole_field_planPLG.fig
%      WHOLE_FIELD_PLANPLG, by itself, creates a new WHOLE_FIELD_PLANPLG or raises the existing
%      singleton*.
%
%      H = WHOLE_FIELD_PLANPLG returns the handle to a new WHOLE_FIELD_PLANPLG or the handle to
%      the existing singleton*.
%
%      WHOLE_FIELD_PLANPLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in WHOLE_FIELD_PLANPLG.M with the given input arguments.
%
%      WHOLE_FIELD_PLANPLG('Property','Value',...) creates a new WHOLE_FIELD_PLANPLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Whole_field_planPLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Whole_field_planPLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Whole_field_planPLG

% Last Modified by GUIDE v2.5 29-Dec-2016 17:01:00

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Whole_field_planPLG_OpeningFcn, ...
                   'gui_OutputFcn',  @Whole_field_planPLG_OutputFcn, ...
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


% --- Executes just before Whole_field_planPLG is made visible.
function Whole_field_planPLG_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for RadiationPlanPLG
handles.output = hObject;
% Add handle of calling object
handles.hh = varargin{1};
handles.Project_name = arbuz_get(handles.hh, 'FILENAME'); %SAVES
Slashes = strfind(handles.Project_name,'\');
handles.Out_path = handles.Project_name(1:Slashes(end));
handles.Experiment_tag =  handles.Project_name(Slashes(end-1)+1:Slashes(end)-1);
Output_fields = {'Name','Filename','SLAVELIST'};
%output_list = arbuz_FindImage(hGUI, input_list, criterion, arg, output_fields)

%Get List of CT Images 
Ct_images_idx = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'DICOM3D', Output_fields);
%Set values of possible CT images for use.
for ii = 1:length(Ct_images_idx)
   Ct_string{ii} =  Ct_images_idx{ii}.Name;    
end
set(handles.CT_pop_up,'String', Ct_string)




% %Get List of MRI images .
% MRI_images = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'MRI', Output_fields);
% %A = arbuz_FindImage(Arbuz_handles, Experiment_list{ii}.T1_MRI_for_Duct, 'FINDSLAVESWITHINNAME', 'duct', Output_fields);
% %Set values of possible CT images for use.
% for ii = 1:length(MRI_images)
%    MRI_images_string{ii} =  MRI_images{ii}.Name;    
% end
% set(handles.MRI_pop_up,'String', MRI_images_string)


% Update handles structure
guidata(hObject, handles);


% UIWAIT makes Whole_field_planPLG wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Whole_field_planPLG_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in CT_pop_up.
function CT_pop_up_Callback(hObject, eventdata, handles)
% hObject    handle to CT_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns CT_pop_up contents as cell array
%        contents{get(hObject,'Value')} returns selected item from CT_pop_up


% --- Executes during object creation, after setting all properties.
function CT_pop_up_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CT_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on selection change in pO2_pop_up.
function pO2_pop_up_Callback(hObject, eventdata, handles)
% hObject    handle to pO2_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pO2_pop_up contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pO2_pop_up
String = get(handles.pO2_pop_up,'String');
Selection = get(handles.pO2_pop_up,'Value');
MRI_name = String{Selection};
Output_fields = {'Name','Filename','SLAVELIST'};
MRI_images = arbuz_FindImage(handles.hh, 'master', 'NAME', MRI_name, Output_fields);
for ii = 1:length(MRI_images{1}.SlaveList)
    Tumor_string{ii} = MRI_images{1}.SlaveList{ii}.SlaveName 
end
set(handles.Tumor_mask_pop_up,'String', Tumor_string)

% Update handles structure
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function pO2_pop_up_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pO2_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
    
    
end


% % --- Executes on selection change in MRI_pop_up.
% function MRI_pop_up_Callback(hObject, eventdata, handles)
% % hObject    handle to MRI_pop_up (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% 
% % Hints: contents = cellstr(get(hObject,'String')) returns MRI_pop_up contents as cell array
% %        contents{get(hObject,'Value')} returns selected item from MRI_pop_up
% String = get(handles.MRI_pop_up,'String');
% Selection = get(handles.MRI_pop_up,'Value');
% MRI_name = String{Selection};
% Output_fields = {'Name','Filename','SLAVELIST'};
% MRI_images = arbuz_FindImage(handles.hh, 'master', 'NAME', MRI_name, Output_fields);
% for ii = 1:length(MRI_images{1}.SlaveList)
%     Tumor_string{ii} = MRI_images{1}.SlaveList{ii}.SlaveName 
% end
% set(handles.Tumor_mask_pop_up,'String', Tumor_string)
% 
% % Update handles structure
% guidata(hObject, handles);
% 
% 
% 
% % --- Executes during object creation, after setting all properties.
% function MRI_pop_up_CreateFcn(hObject, eventdata, handles)
% % hObject    handle to MRI_pop_up (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    empty - handles not created until after all CreateFcns called
% 
% % Hint: popupmenu controls usually have a white background on Windows.
% %       See ISPC and COMPUTER.
% if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
%     set(hObject,'BackgroundColor','white');
% end


% --- Executes on selection change in Tumor_mask_pop_up.
function Tumor_mask_pop_up_Callback(hObject, eventdata, handles)
% hObject    handle to Tumor_mask_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hints: contents = cellstr(get(hObject,'String')) returns Tumor_mask_pop_up contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Tumor_mask_pop_up


% --- Executes during object creation, after setting all properties.
function Tumor_mask_pop_up_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Tumor_mask_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Create_Beams.
function Create_Beams_Callback(hObject, eventdata, handles)
% hObject    handle to Create_Beams (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%Assumption that images are unquiely named.
%
%Code to produce the plugs goes here:
%

%Basic logic of how the plugs are produced:
%1)Pull imaging information from from project acording to user input from the plugin
%GUI.


% -Matt Maggio 12/29/2016


%Start by Gathering Information from the Images+
Output_fields = {'Name','Filename','SLAVELIST','AShow','data'};

%Gather CT info
String_CT = get(handles.CT_pop_up,'String');
Selection_CT = get(handles.CT_pop_up,'Value');
CT_name = String_CT{Selection_CT};
CT_image_info = arbuz_FindImage(handles.hh, 'master', 'NAME', CT_name, Output_fields);
CT_data = CT_image_info{1}.data;
CT_transformation = CT_image_info{1}.Ashow;

CT = struct('data',CT_data , 'name', CT_name , 'info' , CT_image_info , 'transform', CT_transformation )


%Copy path for output. Should output to the same path as the registration
%project.
Experiment_name =  handles.Experiment_tag;
Experiment_path = handles.Out_path;
clear prj

%Standard angle sets. the angles we input here create the bev masks.
%however they don't corrispond exactly to the XRad gantry. So we have to
%flip them if they don't lie between 0 and 90;
%angles = 0:360/5:360 -(360/5); %standard 5 port treatments. 
angles = 90 : -72 :-270+72; %NEW Standard 5 port treatment for WL Validation.
%angles(1) = 0; %replace angle 1 with 0 to check geometry 

% over_90 = angles>90;
% Gantry_angles = angles - 360*over_90 %Historical code. Leave it.

%pile all information into a struct. This will get saved with the plug
%production dataset so we can allways reproduce the plugs.
plan_param = struct('Find_skin_method', handles.Find_skin_method.String{handles.Find_skin_method.Value}, 'Field_size_mm', 35,...
    'Experiment_name' ,Experiment_name ,'Experiment_path', Experiment_path, 'prescription' , str2num(handles.Prescription_edit.String), 'Plan_plugs' , 0,...
    'Output_dataset_name', 'Whole_field_plan_dataset', 'Post_process' , 0,'Output_ini',1,'Conformal_method','Whole field treatment APPA');
 
%  plan_param.Gantry_angles = 90 : -72 :-270+72; 
 

CT_Frame_data.CT = CT ;

[ Planning_Output ] = Generalized_Dose_planning_func( CT_Frame_data , plan_param );





  
    

    




% --- Executes on button press in UVcheckbox1.
function UVcheckbox1_Callback(hObject, eventdata, handles)
% hObject    handle to UVcheckbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of UVcheckbox1


% --- Executes on selection change in Boost_anti_boost_popup.
function Boost_anti_boost_popup_Callback(hObject, eventdata, handles)
% hObject    handle to Boost_anti_boost_popup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns Boost_anti_boost_popup contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Boost_anti_boost_popup


% --- Executes during object creation, after setting all properties.
function Boost_anti_boost_popup_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Boost_anti_boost_popup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Margin_input_box_Callback(hObject, eventdata, handles)
% hObject    handle to Margin_input_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Margin_input_box as text
%        str2double(get(hObject,'String')) returns contents of Margin_input_box as a double


% --- Executes during object creation, after setting all properties.
function Margin_input_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Margin_input_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Construct_coverage_map.
function Construct_coverage_map_Callback(hObject, eventdata, handles)
% hObject    handle to Construct_coverage_map (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Construct_coverage_map


% --- Executes on button press in Manual_thresh.
function Manual_thresh_Callback(hObject, eventdata, handles)
% hObject    handle to Manual_thresh (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Manual_thresh


% --- Executes on selection change in Plug_holder_size.
function Plug_holder_size_Callback(hObject, eventdata, handles)
% hObject    handle to Plug_holder_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns Plug_holder_size contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Plug_holder_size


% --- Executes during object creation, after setting all properties.
function Plug_holder_size_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Plug_holder_size (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in Operation_from_user.
function Operation_from_user_Callback(hObject, eventdata, handles)
% hObject    handle to Operation_from_user (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns Operation_from_user contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Operation_from_user


% --- Executes during object creation, after setting all properties.
function Operation_from_user_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Operation_from_user (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in Find_skin_method.
function Find_skin_method_Callback(hObject, eventdata, handles)
% hObject    handle to Find_skin_method (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns Find_skin_method contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Find_skin_method


% --- Executes during object creation, after setting all properties.
function Find_skin_method_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Find_skin_method (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in Plug_production_method.
function Plug_production_method_Callback(hObject, eventdata, handles)
% hObject    handle to Plug_production_method (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns Plug_production_method contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Plug_production_method


% --- Executes during object creation, after setting all properties.
function Plug_production_method_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Plug_production_method (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Prescription_edit_Callback(hObject, eventdata, handles)
% hObject    handle to Prescription_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Prescription_edit as text
%        str2double(get(hObject,'String')) returns contents of Prescription_edit as a double


% --- Executes during object creation, after setting all properties.
function Prescription_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Prescription_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
