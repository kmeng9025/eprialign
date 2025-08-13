function varargout = Project_to_plugs_PLG(varargin)
% PROJECT_TO_PLUGS_PLG MATLAB code for Project_to_plugs_PLG.fig
%      PROJECT_TO_PLUGS_PLG, by itself, creates a new PROJECT_TO_PLUGS_PLG or raises the existing
%      singleton*.
%
%      H = PROJECT_TO_PLUGS_PLG returns the handle to a new PROJECT_TO_PLUGS_PLG or the handle to
%      the existing singleton*.
%
%      PROJECT_TO_PLUGS_PLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PROJECT_TO_PLUGS_PLG.M with the given input arguments.
%
%      PROJECT_TO_PLUGS_PLG('Property','Value',...) creates a new PROJECT_TO_PLUGS_PLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Project_to_plugs_PLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Project_to_plugs_PLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Project_to_plugs_PLG

% Last Modified by GUIDE v2.5 10-Aug-2016 13:18:21

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Project_to_plugs_PLG_OpeningFcn, ...
                   'gui_OutputFcn',  @Project_to_plugs_PLG_OutputFcn, ...
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


% --- Executes just before Project_to_plugs_PLG is made visible.
function Project_to_plugs_PLG_OpeningFcn(hObject, eventdata, handles, varargin)
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

%Get List of pO2 images
pO2_images_idx = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'PO2_pEPRI', Output_fields);
%Set values of possible pO2 images for use.
for ii = 1:length(pO2_images_idx)
   pO2_string{ii} =  pO2_images_idx{ii}.Name;    
end
set(handles.pO2_pop_up,'String', pO2_string)


%Get List of MRI images .
MRI_images = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'MRI', Output_fields);
%A = arbuz_FindImage(Arbuz_handles, Experiment_list{ii}.T1_MRI_for_Duct, 'FINDSLAVESWITHINNAME', 'duct', Output_fields);
%Set values of possible CT images for use.
for ii = 1:length(MRI_images)
   MRI_images_string{ii} =  MRI_images{ii}.Name;    
end
set(handles.MRI_pop_up,'String', MRI_images_string)


% Update handles structure
guidata(hObject, handles);


% UIWAIT makes Project_to_plugs_PLG wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Project_to_plugs_PLG_OutputFcn(hObject, eventdata, handles) 
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


% --- Executes on selection change in MRI_pop_up.
function MRI_pop_up_Callback(hObject, eventdata, handles)
% hObject    handle to MRI_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns MRI_pop_up contents as cell array
%        contents{get(hObject,'Value')} returns selected item from MRI_pop_up
String = get(handles.MRI_pop_up,'String');
Selection = get(handles.MRI_pop_up,'Value');
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
function MRI_pop_up_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MRI_pop_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


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


% --- Executes on button press in Create_Plugs.
function Create_Plugs_Callback(hObject, eventdata, handles)
% hObject    handle to Create_Plugs (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%Assumption that images are unquiely named.
%
%Code to produce the plugs goes here:
%
%Start by Gathering Information from the Images+
Output_fields = {'Name','Filename','SLAVELIST','AShow'};
prj = load(handles.Project_name);


%Gather CT info
String_CT = get(handles.CT_pop_up,'String');
Selection_CT = get(handles.CT_pop_up,'Value');
CT_name = String_CT{Selection_CT};
CT_image_info = arbuz_FindImage(handles.hh, 'master', 'NAME', CT_name, Output_fields);
CT = prj.images{CT_image_info{1}.ImageIdx}.data;
CT_transformation = CT_image_info{1}.Ashow;

%Gather EPRI info
String_EPRI = get(handles.pO2_pop_up,'String');
Selection_EPRI = get(handles.pO2_pop_up,'Value');
EPRI_name = String_EPRI{Selection_EPRI};
EPRI_image_info = arbuz_FindImage(handles.hh, 'master', 'NAME', EPRI_name, Output_fields);
EPRI = prj.images{EPRI_image_info{1}.ImageIdx}.data;
EPRI_transformation = EPRI_image_info{1}.Ashow;

%Gather MRI info
String_MRI = get(handles.MRI_pop_up,'String');
Selection_MRI = get(handles.MRI_pop_up,'Value');
MRI_name = String_MRI{Selection_MRI};
MRI_image_info = arbuz_FindImage(handles.hh, 'master', 'NAME', MRI_name, Output_fields);
MRI = prj.images{MRI_image_info{1}.ImageIdx}.data;
MRI_transformation = MRI_image_info{1}.Ashow;

String_Tumor = get(handles.Tumor_mask_pop_up,'String');
Selection_Tumor = get(handles.Tumor_mask_pop_up,'Value');
Tumor_name = String_Tumor{Selection_Tumor};
Tumor_image_info = arbuz_FindImage(handles.hh, MRI_image_info, 'FINDSLAVESWITHINNAME', Tumor_name, Output_fields);
for ii = 1:length(Tumor_image_info)
   if strfind(Tumor_image_info{ii}.ImageType,'3DMASK')
       Tumor_image_info{1}.SlaveIdx = Tumor_image_info{ii}.SlaveIdx;
   end
end
Tumor_mask = prj.images{MRI_image_info{1}.ImageIdx}.slaves{Tumor_image_info{1}.SlaveIdx}.data;
cut_off = 0.5 ;

Time_all= tic %time all the time consuming stuff

%Transform the MRI Tumor mask into the CT
disp('tranforming tumor to CT')
tic
Tumor_CT = reslice_volume(inv(CT_transformation),inv(MRI_transformation),zeros(size(CT)), double(Tumor_mask) , 0, 1) > cut_off;
toc
%
%Mask_out = reslice_volume(inv(Transform_to),inv(Transform_from),zeros(Dims_to), Mask to be transformed , 0, 1) > cut_off;
%    new_image.data = reslice_volume(inv(find_dest3D{1}.Ashow), inv(find_mask3D{ii}.Ashow),zeros(Dims), preprocessed_data, 0, 1) > cut_off;

%Transform the EPR Hypoxia mask into the CT
Hypoxia_mask =zeros(size(EPRI));
Hypoxia_mask(find(EPRI<=10 & EPRI~=-100 )) =1;
disp('tranforming Hypoxia to CT')
tic
Hypoxia_CT = reslice_volume(inv(CT_transformation),inv(EPRI_transformation),zeros(size(CT)), double(Hypoxia_mask) , 0, 1) > cut_off;
toc

%Transform the EPR into the CT
% disp('tranforming EPRI to CT')
% tic
% EPRI_CT = reslice_volume(inv(CT_transformation),inv(EPRI_transformation),zeros(size(CT)), double(EPRI) , 0, 1) > cut_off;
% toc


%Transform the MRI Tumor mask into the EPRI
disp('tranforming tumor to EPRI')
tic
Tumor_EPRI = reslice_volume(inv(EPRI_transformation),inv(MRI_transformation),zeros(size(EPRI)), double(Tumor_mask) , 0, 1) > cut_off;
toc


% 
%Generate Bev_masks
%  load('Z:\CenterProjects\Example_dir\CT_hypoxia_mask.mat')
%  load('Z:\CenterProjects\Example_dir\CT_tumor_mask.mat')
%  load('Z:\CenterProjects\Example_dir\CT_boost_mask.mat')
%  load('Z:\CenterProjects\Example_dir\CT_file.mat')

%addpath('Z:\CenterProjects\IMRT_3D_print_connector')

%[ Test_obj , Sphere_mask , Sphere_mask_large] = Generate_test_obj(  );
%Flag for boost vs antiboost
mask_out_boost= true;

%Standard angle sets. the angles we input here create the bev masks.
%however they don't corrispond exactly to the XRad gantry. So we have to
%flip them if they don't lie between 0 and 90;
%angles = 0:360/5:360 -(360/5); %standard 5 port treatments. 
angles = 90 : -72 :-270+72; %NEW Standard 5 port treatment for WL Validation.
%angles(1) = 0; %replace angle 1 with 0 to check geometry 

over_90 = angles>90;
Gantry_angles = angles - 360*over_90

%Find the W_L values before we get into the BEVS. %Old Code     
%  UV_table_path = 'Z:\CenterProjects\IMRT_3_development\UV_plug_shift_table\UV_plug_shift_table';
%  load(UV_table_path)
%  WL_shift = {};
%  for ii = 1:length(Gantry_angles)   
%     WL_idx = find(abs(Angle_idx-Gantry_angles(ii) ) == min(abs(Angle_idx-Gantry_angles(ii))));
%     WL_shift{ii} =  UV_plug_shift_table{WL_idx};
%  end
%      
 %Find the W_L values before we get into the BEVS.
Table_name = 'Z:\CenterProjects\IMRT_3_development\Plug_Shift_table\Final_plug_shift_table.xlsx' ;
W_L_table = xlsread(Table_name);
for ii= 1:5
% WL_shift{ii}.Plug_X = W_L_table(ii,4) + W_L_table(ii,7); 
% WL_shift{ii}.Plug_Y = W_L_table(ii,5) + W_L_table(ii,8);
WL_shift{ii}.Plug_X = W_L_table(ii,4);
WL_shift{ii}.Plug_Y = W_L_table(ii,5);
end

 
%figure(1); clf; 
%ibGUI(CT_hypoxia_mask)
%isosurface((CT_Tumor_mask&CT_hypoxia_mask),0.5)

%Change the target from Hypoxia to 3 Offset spheres 3mm in radius. Comment out for a real experiment.  
%CT_target_1  = Create_3D_target_volume_3D_phantom_3_sphere( CT ,6.5, [0 0 0] );
% CT_target_1  = Create_3D_target_volume_3D_phantom_3_sphere( CT ,2, [0 0 0] );
% CT_target_2  = Create_3D_target_volume_3D_phantom_3_sphere( CT ,2 , [4 0 4] );
% CT_target_3  = Create_3D_target_volume_3D_phantom_3_sphere( CT ,2 , [0 4 -4] );
% Hypoxia_CT = CT*0;
%  Hypoxia_CT(find(CT_target_1==1 | CT_target_2==1 | CT_target_3==1 ))= 1;
%   Tumor_CT = Hypoxia_CT;
%  ibGUI(Hypoxia_CT)
% %Chuck's code for BEV mask production
% %[Bev_masks,  total_coverage ] = maskbev_with_3D_beams(angles , Tumor_CT, Hypoxia_CT, mask_out_boost  );

%Determine the stage shift required to satisfy the Bevmask assumption that
%the center of the target is at the isocenter of the treatment field.

%Target= Tumor_CT & Hypoxia_CT; % Changed 8/8/2016 to only target based on
%hypoxia

Target= Hypoxia_CT;

goodpix=find(Target); 
[igood,jgood,kgood]=ind2sub(size(Target),goodpix);
%These 3 lines center the BEV field on the center of mass of the target.
icen=mean(igood); %old code for CENTER OF MASS
jcen=mean(jgood);
kcen=mean(kgood);
imid = (size(Target,1)/2);
jmid = (size(Target,2)/2);
kmid = (size(Target,3)/2);


% Generate Bevs for the target volume (hypoxia_CT) and for the tumor
% itself
 [Bev_masks] = maskbev(angles , Tumor_CT, Hypoxia_CT, mask_out_boost  )
% [Bev_masks_Tumor] = maskbev(angles , Tumor_CT, Tumor_CT, mask_out_boost  )

% % 
%  %Generate Cumluative sum histogram for treatment assesment. Returns figure
%  %handles. 
%  Target = Hypoxia_CT & Tumor_CT;
%  Non_target = ~Target & Tumor_CT;
%  [ H ] = Cum_sum_hist( Target, Non_target, Coverage_mask )
%  
%  %Figure out the treatment based on the old algorithim. For compare. 
%  %[isocenter, res] = TargetHypoxicSphere2(ProxyList{1}.data,[0.6629,0.6629,0.6629],TumorMask, handles.DBOOST(ii)/2);
%  Boost_options = [5 6.75 7.5 ];
%  clear isocenter res hit_vol
%  for ii = 1:length(Boost_options)
%  [isocenter{ii}, res{ii}] = TargetHypoxicSphere2(EPRI,[0.6629,0.6629,0.6629],Tumor_EPRI, Boost_options(ii)/2);
%  hit_vol(ii) = res{ii}.hit_volume;
%  end
%  tumor_vol = numel(find(Tumor_EPRI))*(0.6629^3)
%  Too_large = ~(hit_vol > (tumor_vol * 0.56)); 
%  hit_vol = hit_vol .* Too_large
%  Isocenter = isocenter{find(max(hit_vol) == hit_vol)};
%  %Diameter = Boost_options(find(max(hit_vol) == hit_vol));
%  Diameter= 6.75
%  [ Spherical_target ] = Create_3D_target_volume_3D_phantom_3_sphere_generalized_1( Tumor_EPRI ,Diameter/2 , [Isocenter(2) Isocenter(1) Isocenter(3)] ,0.6629 );
%  %Transform the EPR Hypoxia mask into the CT
%     disp('tranforming sphere into CT')
%     tic
%     Sphere_CT = reslice_volume(inv(CT_transformation),inv(EPRI_transformation),zeros(size(CT)), double(Spherical_target) , 0, 1) > cut_off;
%     toc
%  %if we feed our target volume into the Bev mask code we'll get centerer
%  %plugs that relate to the arc boost radiation. lets try it.
%  [Bev_masks_spherical] = maskbev(angles , Tumor_CT, Sphere_CT, mask_out_boost  )
%  
%   Coverage_mask_spherical = Tumor_CT*0;
%  for ii = 1:length(Bev_masks)
%  [ Bev_masks{ii}.Coverage_mask ] = mask_bev_quant(Tumor_CT, Sphere_CT , Bev_masks_spherical{ii}.Boost_map, Bev_masks{ii}.Angle);
%    Coverage_mask_spherical = Coverage_mask_spherical + Bev_masks{ii}.Coverage_mask ;
%  end
% 
%  [ H2 ] = Cum_sum_hist( Target, Non_target, Coverage_mask_spherical )
%  
 


Experiment_details.Experiment_name =  handles.Experiment_tag;
Experiment_details.Experiment_path = handles.Out_path;

Experiment_details.Bed_shift_textfile = sprintf('%s',handles.Out_path, handles.Experiment_tag,'_Bed_shift.txt');
fid = fopen(Experiment_details.Bed_shift_textfile, 'w+');
%Writes the bed shift to a text file. Important to note that X= j and Y = i This is the classic problem with Matlab having colums be the first dim.  
str = sprintf('%s', ' Move the bed by  X  ', num2str((jcen - jmid)/-100),'   Y  ', num2str((icen - imid)/100), '    Z    ', num2str((kcen - kmid)/100)); 
fprintf(fid, str);
fclose('all');

%After bev mask productions. Produce Bev_masks dilated by some ammount
Margin = str2num(get(handles.Margin_input_box,'string')); %margin in mm
SE = strel('disk',floor(Margin/0.025));
SE_1 = strel('disk',floor((0.3)/0.025)); %Just erode the tumor mask by 0.3mm to account for printer shrinking.
        %the antiboost. Need to justify before doing.
%SE_2 = strel('disk',floor((Margin/1.5)/0.025));  

parfor ii = 1:length(Bev_masks)
    
        %Represents an erosion by 0.1mm or 4 smallpix
        %SE = ones(floor((Margin/0.025)),floor((Margin/0.025)))
        %Target = imdilate(Target,SE);
        Bev_masks{ii}.Dilated_boost_map = imdilate(Bev_masks{ii}.Boost_map,SE);
        %Bev_masks{ii}.Double_Dilated_boost_map = imdilate(Bev_masks{ii}.Dilated_boost_map,SE);
        Bev_masks{ii}.Margin_dilated = Margin;
        [ Bev_masks{ii} ] = Find_antiboost_2D_mask_3( Bev_masks{ii}, Bev_masks_Tumor{ii}   )
        
        %Dilate the antiboost mask. Make sure to resubtract the boost mask      
        Bev_masks{ii}.Dilated_Anti_boost_map = imdilate(Bev_masks{ii}.Antiboost_map,SE_1);
        Bev_masks{ii}.Dilated_Anti_boost_map((Bev_masks{ii}.Dilated_boost_map==1))=0;
        %Bev_masks{ii}.Dilated_Anti_boost_map(find(imdilate(Bev_masks{ii}.Boost_map,SE_1)==1))=0;
        %Figure the support mask to keep the dilated antiboost map from
        %being unprintable. 
        
        %We could just Print a full layer because this is dumb to fight with.
%         Bev_masks{ii}.Support_mask = Bev_masks{ii}.Dilated_boost_map + Bev_masks{ii}.Dilated_Anti_boost_map ;
%         [ Bev_masks{ii}.Support_mask ] = Add_support_to_mask( Bev_masks{ii} )
%          Bev_masks{ii}.Support_mask  = imdilate(Bev_masks{ii}.Support_mask,SE_2); 
        
%          figure(1); imagesc(Bev_masks{ii}.Boost_map);
%          figure(2); imagesc(Bev_masks{ii}.Dilated_boost_map);
%          figure(3); imagesc(Bev_masks{ii}.Antiboost_map);
%          figure(4); imagesc(Bev_masks{ii}.Dilated_Anti_boost_map);
%          pause(1.5)    
end

if handles.Construct_coverage_map.Value ==1
%get coverage mask for the boost
 parfor ii = 1:length(Bev_masks)
 [ Bev_masks{ii}.Coverage_mask_Boost ] = mask_bev_quant(Tumor_CT, Hypoxia_CT , Bev_masks{ii}.Dilated_boost_map, Bev_masks{ii}.Angle);  
 end
 
  %get coverage mask for the boost
 parfor ii = 1:length(Bev_masks)
 [ Bev_masks{ii}.Coverage_mask_AntiBoost ] = mask_bev_quant(Tumor_CT, Hypoxia_CT , Bev_masks{ii}.Dilated_Anti_boost_map, Bev_masks{ii}.Angle);  
 end
end



%Save those images in a workspace file 
 save(sprintf('%s',handles.Out_path,'Plug_production_dataset'),'Tumor_CT','Hypoxia_CT', 'CT', 'Bev_masks','WL_shift','Experiment_details' ,'-v7.3')
 
% 
% switch  handles.Boost_anti_boost_popup.String{handles.Boost_anti_boost_popup.Value}
%     case {'Boost' }

switch handles.Boost_anti_boost_popup.String{handles.Boost_anti_boost_popup.Value}
    case 'Boost'



disp('Creating Boost Plug files')
 for ii = 1:length(Bev_masks) 
     
     stl_file = sprintf('%s',handles.Out_path, handles.Experiment_tag ,'_Boost','_',num2str(ii),'_',num2str(Bev_masks{ii}.Angle),'.stl');
     scad_file = sprintf('%s',handles.Out_path, handles.Experiment_tag ,'_Boost','_',num2str(ii),'_',num2str(Bev_masks{ii}.Angle),'.scad');
     Bev_masks{ii}.stl_filename = stl_file;
     Bev_masks{ii}.scad_filename = scad_file;   
        
        Target = Bev_masks{ii}.Dilated_boost_map;

        %imagesc(Target_dil)

        [Bound_regions,L] = bwboundaries(Target,'noholes');
        
        %imshow(BW)
        %hold on
        clear Poly_vector
        for k = 1:length(Bound_regions)
           boundary = Bound_regions{k};
           %plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
        
           Poly_vector{k}(:,1) =  boundary(:,1) - size(Target, 1)/2;
           Poly_vector{k}(:,2) =  boundary(:,2) - size(Target, 2)/2;
        end
        
        %Scale_factor = [0.05, 0.05, 1];
        Scale_factor = [0.025, 0.025, 1];
        %Scale_factor from chucks bev program Chuck has the pix set to
        %0.025mm. Scad expects everything to be in 1mm units. Therefore we
        %have to shrink the incoming objects by 1*0.025 Z doesnt matter
        %here as cuts should go through the block.
        
    %epr_Contour2OpenSCAD(mod_name, fname, vectors, scale)
    %epr_Contour2OpenSCAD_Plugs(mod_name, fname, vectors, scale)
    %Boris code to write the scad file with my scad code for the plug
    %appended on it.
    %Generate Openscad files
    
    %Get the Winston Lutz (UV shifts in the plug positon from a table)
    Do_UV_shift = get(handles.UVcheckbox1,'Value')
    if Do_UV_shift
        
%     UV_table_path = 'Z:\CenterProjects\IMRT_3_development\UV_plug_shift_table\UV_plug_shift_table';
%     load(UV_table_path)
    UVShift = [WL_shift{ii}.Plug_X WL_shift{ii}.Plug_Y];
    %UVShift = [5 0] %in mm
    else
        UVShift = [0 0]; 
    end
    disp('Writing SCAD files and rendering plugs for Boost')
        epr_Contour2OpenSCAD_Plugs('Plug_cut', ...
          scad_file,...
          Poly_vector,...
          Scale_factor, UVShift);
      
      Openscad_command_path = 'Z:\CenterHardware\3Dprint\OpenSCAD';
      Openscad_command_str = 'openscad';
      
      %Render plugs
      cd(Openscad_command_path) %Cd to the sharedrive openscad location.       
      Command_string = sprintf('%s',Openscad_command_str,' -o ',Bev_masks{ii}.stl_filename,' ', Bev_masks{ii}.scad_filename);      
      [status , result] = dos(Command_string)
      
 end
        
 
 
      %Last step is to output an INI file to the outpath for loading into the pilot
      %software.
	  presciption = 13;
      if handles.Manual_thresh.Value ==1
      [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_mask_inputs_manual(CT, Tumor_CT, Hypoxia_CT, Bev_masks, angles, varargin)
      else          
       [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_mask_inputs(CT, Tumor_CT, Hypoxia_CT, Bev_masks, angles, presciption)
      end
      Beam_plan_INI_write_V3(handles.Out_path, handles.Experiment_tag ,Gantry_angles,presciption/length(Gantry_angles), beamtimes);

      

 

    case 'AntiBoost'

 disp('Creating Anti Boost Plug files')
 for ii = 1:length(Bev_masks)     
     stl_file = sprintf('%s',handles.Out_path, handles.Experiment_tag ,'_Anti_Boost','_',num2str(ii),'_',num2str(Bev_masks{ii}.Angle),'.stl');
     scad_file = sprintf('%s',handles.Out_path, handles.Experiment_tag ,'_Anti_Boost','_',num2str(ii),'_',num2str(Bev_masks{ii}.Angle),'.scad');
     Bev_masks{ii}.stl_filename = stl_file;
     Bev_masks{ii}.scad_filename = scad_file;   
        
        Target = Bev_masks{ii}.Dilated_Anti_boost_map;
        %Support = Bev_masks{ii}.Support_mask;

        %imagesc(Target_dil)

        [Bound_regions,L] = bwboundaries(Target,'holes');
        %[Bound_regions_support,L] = bwboundaries(Support,'holes');
        
        %imshow(BW)
        %hold on
        clear Poly_vector  %Convert to centered rather than vectored input
        for k = 1:length(Bound_regions)
           boundary = Bound_regions{k};
           %plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
        
           Poly_vector{k}(:,1) =  boundary(:,1) - size(Target, 1)/2;
           Poly_vector{k}(:,2) =  boundary(:,2) - size(Target, 2)/2;
        end
        
        %investigate the Poly_vector of the mask. Identify any outlines
        %which are internal. These need to be classified differenly so they are added to the plug instead of subtracted.
        clear Poly_vectors_struct
        Poly_vectors_struct = {};
        for k = 1:length(Bound_regions)
        Outline = zeros(size(Target));
        idx = sub2ind(size(Target),Bound_regions{k}(:,1),Bound_regions{k}(:,2));
        Outline(idx) = 1;
        Outline_filled = imfill(Outline,'holes');
        if numel(find(Target(find(Outline_filled-Outline))))>1;
        disp('outside border')
        else
        disp('inside border adding to structure')
        Poly_vectors_struct{end+1} = Bound_regions{k};
        end
        end
        
        %figure; imagesc(Target); hold on; for vect = 1:length(Poly_vectors_struct); plot(Poly_vectors_struct{vect}(:,2),Poly_vectors_struct{vect}(:,1), 'm');end
        
        clear Poly_vectors_structure %Convert to centered rather than vectored input
        if length(Poly_vectors_struct)> 0
        for k = 1:length(Poly_vectors_struct)
           boundary = Poly_vectors_struct{k};
           %plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)        
           Poly_vectors_structure{k}(:,1) =  boundary(:,1) - size(Target, 1)/2;
           Poly_vectors_structure{k}(:,2) =  boundary(:,2) - size(Target, 2)/2;
        end
        else
           Poly_vectors_structure = {[0 0]};
        end
         
%        
%         clear Poly_vector_support  %Convert to centered rather than vectored input
%         if length(Bound_regions_support)>0
%         for k = 1:length(Bound_regions_support)
%            boundary_support = Bound_regions_support{k};
%            %plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)        
%            Poly_vector_support{k}(:,1) =  boundary_support(:,1) - size(Support, 1)/2;
%            Poly_vector_support{k}(:,2) =  boundary_support(:,2) - size(Support, 2)/2;
%         end
%         else
%             Poly_vector_support = {[0 0]};
%         end
        
        
        %Scale_factor = [0.05, 0.05, 1];
        Scale_factor = [0.025, 0.025, 1];
        %Scale_factor from chucks bev program Chuck has the pix set to
        %0.025mm. Scad expects everything to be in 1mm units. Therefore we
        %have to shrink the incoming objects by 1*0.025 Z doesnt matter
        %here as cuts should go through the block.
        
    %epr_Contour2OpenSCAD(mod_name, fname, vectors, scale)
    %epr_Contour2OpenSCAD_Plugs(mod_name, fname, vectors, scale)
    %Boris code to write the scad file with my scad code for the plug
    %appended on it.
    %Generate Openscad files
    
    %Get the Winston Lutz (UV shifts in the plug positon from a table)
    Do_UV_shift = get(handles.UVcheckbox1,'Value')
    if Do_UV_shift
        
%     UV_table_path = 'Z:\CenterProjects\IMRT_3_development\UV_plug_shift_table\UV_plug_shift_table';
%     load(UV_table_path)
    UVShift = [WL_shift{ii}.Plug_X WL_shift{ii}.Plug_Y]; 
    %UVShift = [5 0] %in mm
    else
        UVShift = [0 0]; 
    end
    disp('Writing SCAD files and rendering plugs for anti_boost')
%         epr_Contour2OpenSCAD_Plugs_Plus_Support('Plug_cut','Plug_structures', ...
%           scad_file,...
%           Poly_vector,...
%           Poly_vectors_structure,...
%           Poly_vector_support,...
%           Scale_factor, UVShift);

      epr_Contour2OpenSCAD_Plugs_1_full_layer_plus_structure('Plug_cut','Plug_structures',...
        scad_file,...
        Poly_vector,Poly_vectors_structure,...
        Scale_factor, UVShift);
      %epr_Contour2OpenSCAD_Plugs_Plus_Support(mod_name,structure_mod_name, fname, vectors,vectors_structure,vectors_support, scale, UVShift)
      
      Openscad_command_path = 'Z:\CenterHardware\3Dprint\OpenSCAD';
      Openscad_command_str = 'openscad';
      
      %Render plugs
      cd(Openscad_command_path) %Cd to the sharedrive openscad location.       
      Command_string = sprintf('%s',Openscad_command_str,' -o ',Bev_masks{ii}.stl_filename,' ', Bev_masks{ii}.scad_filename);      
      [status , result] = dos(Command_string)
 end
 
      %Last step is to output an INI file to the outpath for loading into the pilot
      %software.
	  presciption = 13;
      if handles.Manual_thresh.Value ==1
      [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_mask_inputs_manual(CT, Tumor_CT, Hypoxia_CT, Bev_masks, angles, varargin)
      else          
       [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_mask_inputs(CT, Tumor_CT, Hypoxia_CT, Bev_masks, angles, presciption)
      end
      Beam_plan_INI_write_V3(handles.Out_path, handles.Experiment_tag ,Gantry_angles,presciption/length(Gantry_angles), beamtimes);
%  end

end
 
 disp(sprintf('%s','Plug production finished. Files are at ', handles.Out_path ))
 
 toc(Time_all)
%

%

%
%
%


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
