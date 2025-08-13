function [ Experiment_list ] = IMRT_post_processing( Experiment_list, parameters )

%This script should reproduce all of the plug production datasets
%correctly as a post-processing dataset. This should fix the problems that 
%come from trying to do foriensics on the plug production dataset.
CellToWrite={'Experiment Name','Boost Tumor Coverage','Antiboost Tumor Coverage','Alternative Antiboost Tumor Coverage'...
   'Boost Hypoxia Coverage','Antiboost Hypoxia Coverage','Alternative Antiboost Hypoxia Coverage'};

for ii = 1:length(Experiment_list)

    
    if strcmp(Experiment_list{ii}.protocol_2,'Survival data') %Do not process censored data
        if strcmp(safeget(Experiment_list{ii},'protocol', 'Full imaging'),'Just Irradiation')
            continue
        end
        %This part should check to see if there is a post production
        %dataset allready and not do it if there is.
%         [fpath, ffile] = fileparts(Experiment_list{ii}.registration);
%         try
%         handles.plugs = load(fullfile(fpath, 'Plug_production_dataset.mat'));
%         catch
%         end 

Output_fields = {'Name','Filename','SLAVELIST','AShow','data'};
hh = figure(100) 
status = arbuz_OpenProject(hh, Experiment_list{ii}.registration)

%Gather MRI info
MRI_image_info = arbuz_FindImage(hh, 'master', 'IMAGETYPE', 'MRI', Output_fields);
MRI_image_info = arbuz_FindImage(hh, MRI_image_info, 'INNAME', 'ax', Output_fields);
MRI_outline = arbuz_FindImage(hh, MRI_image_info, 'FINDSLAVESWITHINNAME', 'outline', Output_fields);
MRI_outline = arbuz_FindImage(hh, MRI_outline, 'FINDSLAVESIMAGETYPE', '3DMASK', Output_fields);

%Gather CT info
CT_image_info = arbuz_FindImage(hh, 'master', 'INNAME', 'CT', Output_fields);
 CT_data = CT_image_info{1}.data;
CT_transformation = CT_image_info{1}.Ashow;

tic
disp('Transforming MRI outline into CT coords')
Outline_CT = reslice_volume(inv(CT_transformation),inv(MRI_outline{1}.Ashow),zeros(size(CT_data)), double(MRI_outline{1}.data) , 0, 1) > 0.5;
toc

CT = struct('data',CT_data , 'name', CT_image_info{1}.Name , 'info' , CT_image_info , 'transform', CT_transformation, 'MRI_outline_CT',Outline_CT)

%Gather EPRI info
EPRI_image_info = arbuz_FindImage(hh, 'master', 'IMAGETYPE', 'PO2_pEPRI', Output_fields);
EPRI_image_info = arbuz_FindImage(hh, EPRI_image_info, 'INNAME', '002', Output_fields);
 EPRI_data = EPRI_image_info{1}.data;
EPRI_transformation = EPRI_image_info{1}.Ashow;

EPRI = struct('data',EPRI_data , 'name', EPRI_image_info{1}.Name , 'info' , EPRI_image_info , 'transform', EPRI_transformation )

%Gather information on the tumor, make sure the user selected a mask and
%not a surface.

% Tumor_names_order = {'out', 'inf', ''}
% 
% for jj = 1:length(Tumor_names_order)
Tumor_image_info = arbuz_FindImage(hh, EPRI_image_info, 'FINDSLAVESWITHINNAME', 'Tumor_out', Output_fields);
if isempty(Tumor_image_info) == 1
Tumor_image_info = arbuz_FindImage(hh, EPRI_image_info, 'FINDSLAVESWITHINNAME', 'Tumor_inf', Output_fields);
if isempty(Tumor_image_info) == 1
    Tumor_image_info = arbuz_FindImage(hh, EPRI_image_info, 'FINDSLAVESWITHINNAME', 'Tumor', Output_fields);
end
end

if length(Tumor_image_info) >1 
    Tumor_image_info = arbuz_FindImage(hh, Tumor_image_info, 'FINDSLAVESIMAGETYPE', '3DMASK', Output_fields);
end
    
Tumor_mask = Tumor_image_info{1}.data;


Tumor = struct('data', Tumor_mask,'name', Tumor_image_info{1}.Name, 'info' , Tumor_image_info{1} ,'transform' ,EPRI_transformation)


Project_name = arbuz_get(hh, 'FILENAME');
Slashes = strfind(Project_name,'\');
Experiment_path = Project_name(1:Slashes(end));
Experiment_name =  Project_name(Slashes(end-1)+1:Slashes(end)-1);

%Configure defaults for parameters.
Experiment_list{ii}.Boost_or_Anti_boost = safeget(Experiment_list{ii},'Boost_or_Anti_boost','Boost');
Experiment_list{ii}.Conformal_margin = safeget(Experiment_list{ii},'Conformal_margin',1.2);
Post_process = safeget(parameters,'Post_process',1);
Output_dataset_name = safeget(parameters,'Output_dataset_name',' ');
Experiment_list{ii}.Conformal_method = safeget( Experiment_list{ii},'Conformal_method','2 Port Opposed Beams');

%generalized planning function. with switches for not doing plug planning.
plan_param = struct('Boost_or_antiboost', Experiment_list{ii}.Boost_or_Anti_boost, 'Boost_margin',Experiment_list{ii}.Conformal_margin,'Antiboost_margin',Experiment_list{ii}.Conformal_margin/2 , 'Operation_from_user', 'Produce Plugs         ' ,...
    'Find_skin_method', ' ', 'Plug_size', ' ','Experiment_name' ,Experiment_name ,'Experiment_path', Experiment_path,...
    'Boost_dose' , 13, 'Plan_plugs' , 0, 'Output_dataset_name', Output_dataset_name,'Post_process' , Post_process);
 
        disp('Using 5 ports spaced by 72 degrees')
        
        plan_param.Gantry_angles = 90 : -72 :-270+72; %NEW Standard 5 port treatment for WL Validation.
        
        
switch Experiment_list{ii}.Conformal_method
    case '5 Port Equal Spacing'
        disp('Using 5 ports spaced by 72 degrees')
        plan_param.Gantry_angles = 90 : -72 :-270+72; %NEW Standard 5 port treatment for WL Validation.
    case '2 Port Opposed Beams'
        disp('Using 2 Ports spaced 180 degrees apart')
        [ Port_Gantry_Angle ] = IMRT_2_opposed_port_choice( CT , EPRI, Tumor, plan_param )
        plan_param.Gantry_angles = [];
        plan_param.Gantry_angles(1) =  Port_Gantry_Angle;
        opposed = Port_Gantry_Angle - 180 ;
        plan_param.Gantry_angles(2) = opposed + ((Port_Gantry_Angle < -90)*360);        
end
 
 
Coverage_stats = IMRT_n_port_planning( CT , EPRI, Tumor, plan_param );
Experiment_list{ii}.Coverage_stats = Coverage_stats;


if ~length(Coverage_stats)<1

%Saving of coverage stats to hedge against interruptions in the run.
CellToWrite{end+1,1}=Experiment_list{ii}.tag;
CellToWrite{end,2}=Experiment_list{ii}.Coverage_stats.Boost.Tumor_CT_Coverage;
CellToWrite{end,3}=Experiment_list{ii}.Coverage_stats.AntiBoost.Tumor_CT_Coverage;
CellToWrite{end,4}=Experiment_list{ii}.Coverage_stats.Alt_AntiBoost.Tumor_CT_Coverage;
CellToWrite{end,5}=Experiment_list{ii}.Coverage_stats.Boost.Hypoxia_CT_Coverage;
CellToWrite{end,6}=Experiment_list{ii}.Coverage_stats.AntiBoost.Hypoxia_CT_Coverage;
CellToWrite{end,7}=Experiment_list{ii}.Coverage_stats.Alt_AntiBoost.Hypoxia_CT_Coverage;

[Status] = epr_excel_save(safeget(parameters,'ExcelOutputFilename',sprintf('%s',parameters.path,filesep,'EXCELOUTPUT.xlsx')) , CellToWrite); 

end

clear CT EPRI Tumor plan_param
    end
end



end

