function [  ] = IMRT_5_port_planning( CT , EPRI, Tumor, plan_param )

%Code to produce the plugs goes here:
%

%Basic logic of how the plugs are produced:
%1)Pull imaging information from from project acording to user input from the plugin
%GUI.

%2)Transform EPR voxels under 10 torr that are inside the "tumor" mask into
%CT space {Older CHUCK/BORIS CODE}

%3)Run chuck's Maskbev code to determine the 2D beamshapes nessasary to hit
%100% of the target {CHUCK CODE}

%4)Dilate the Maskbev output by the "margin" from the GUI we have
%determined that 1.2mm is our "Safe margin" 

%4a) [optional] reproject the beams through the CT image to show the overlapping beams in 3 space {CHUCK CODE} 

%4b) Bev masks are created with the assumption that the center of mass of
%the target will be at the isocenter of the treatment. To that end we write
%a quick text file that gives the bed shift corrdinates. 
%The bed must be moved this much to statisfy the assumption. While this is
%a bit more complicated it is well worth it in printer time.

%5)Determine if experiment is boost or antiboost. There is a switch to
%excute correct code block. 

%6)Write openscad files using matlab to just write a normal text file and
%name it with a .scad extention.

%7)Call openscad through the commandline matlab interface and cause
%openscad to render the .scad text file into a .stl file. The .stl file
%goes to the printer.

%8)Lastly we write a .ini file for import into the Xrad-225 machine. This file
%contains the instructions for the beams, with gantry angles and beamtime.
%The beamtime is very inportant, so we use {CHUCK CODE} to estimate the
%realistic depth of our object and the realistic dose rate of the
%collimator.

%Simple right? -Matt Maggio 8/22/2016

cut_off = 0.5 ;

Time_all= tic %time all the time consuming stuff

%Transform the MRI Tumor mask into the CT
disp('tranforming tumor to CT')
tic
Tumor_CT = reslice_volume(inv(CT.transform),inv(Tumor.transform),zeros(size(CT.data)), double(Tumor.data) , 0, 1) > cut_off;
toc
%
%Mask_out = reslice_volume(inv(Transform_to),inv(Transform_from),zeros(Dims_to), Mask to be transformed , 0, 1) > cut_off;
%    new_image.data = reslice_volume(inv(find_dest3D{1}.Ashow), inv(find_mask3D{ii}.Ashow),zeros(Dims), preprocessed_data, 0, 1) > cut_off;

%Transform the EPR Hypoxia mask into the CT
Hypoxia_mask =zeros(size(EPRI.data));
Hypoxia_mask(find(EPRI.data<=10 & EPRI.data~=-100 )) =1;
disp('tranforming Hypoxia to CT')
tic
Hypoxia_CT = reslice_volume(inv(CT.transform),inv(EPRI.transform),zeros(size(CT.data)), double(Hypoxia_mask) , 0, 1) > cut_off;
toc

%Transform the EPR into the CT
disp('tranforming EPRI to CT')
tic
EPRI_CT = reslice_volume(inv(CT.transform),inv(EPRI.transform),zeros(size(CT.data)), double(EPRI.data) , 0, 1) ;
toc



%Transform the MRI Tumor mask into the EPRI
% disp('tranforming tumor to EPRI')
% tic
% Tumor_EPRI = reslice_volume(inv(EPRI_transformation),inv(MRI_transformation),zeros(size(EPRI)), double(Tumor.data) , 0, 1) > cut_off;
% toc



%Flag for boost vs antiboost Obsolete, but just leave it because it might
%break something.
mask_out_boost= true;

% %Standard angle sets. the angles we input here create the bev masks.
% %however they don't corrispond exactly to the XRad gantry. So we have to
% %flip them if they don't lie between 0 and 90;
% %angles = 0:360/5:360 -(360/5); %standard 5 port treatments. 
% angles = 90 : -72 :-270+72; %NEW Standard 5 port treatment for WL Validation.
% %angles(1) = 0; %replace angle 1 with 0 to check geometry 
% 
% over_90 = angles>90;
% Gantry_angles = angles - 360*over_90 %Historical code. Leave it.
      
%Find the W_L values before we get into the BEVS.
Table_name = 'Z:\CenterProjects\IMRT_3_development\Plug_Shift_table\Final_plug_shift_table.xlsx' ;
W_L_table = xlsread(Table_name);
for ii= 1:5
% WL_shift{ii}.Plug_X = W_L_table(ii,4) + W_L_table(ii,7); 
% WL_shift{ii}.Plug_Y = W_L_table(ii,5) + W_L_table(ii,8);
WL_shift{ii}.Plug_X = W_L_table(ii,4);
WL_shift{ii}.Plug_Y = W_L_table(ii,5);
end

 
Target= Tumor_CT & Hypoxia_CT; %Target is the intersection of new tumor mask and hypoxia

%Determine the stage shift required to satisfy the Bevmask assumption that
%the center of the target is at the isocenter of the treatment field.
goodpix=find(Target); 
[igood,jgood,kgood]=ind2sub(size(Target),goodpix);
%These 3 lines center the BEV field on the center of mass of the target.
icen=mean(igood); %code for CENTER OF MASS
jcen=mean(jgood);
kcen=mean(kgood);
imid = (size(Target,1)/2);
jmid = (size(Target,2)/2);
kmid = (size(Target,3)/2);

delete(gcp('nocreate'))
%init Parallel computing 
parpool(feature('numCores'))

disp('Generate Bevs for the target volume')
%(hypoxia_CT) 
 [Bev_masks] = maskbev( plan_param.Gantry_angles , Tumor_CT, Hypoxia_CT, mask_out_boost  );
% [Bev_masks_Tumor] = maskbev(angles , Tumor_CT, Tumor_CT, mask_out_boost  )

% sad=307;
% scd=234+10;

%Writes the bed shift to a text file. Important to note that X= j and Y = i This is the classic problem with Matlab having colums be the first dim.  
plan_param.Bed_shift_textfile = sprintf('%s',plan_param.Experiment_path, plan_param.Experiment_name,'_Bed_shift.txt');
fid = fopen(plan_param.Bed_shift_textfile, 'w+');
str = sprintf('%s', ' Move the bed by  X  ', num2str((jcen - jmid)/-100),'   Y  ', num2str((icen - imid)/100), '    Z    ', num2str((kcen - kmid)/100)); 
fprintf(fid, str);
fclose('all');

%IMRT3_margins
% plan_param.Boost_margin = 2.4;
% disp('setting Boost margin to 2.4')
% plan_param.Antiboost_margin = 1.2;
% disp('setting AntiBoost margin to 1.2')
% 
% SE = strel('disk',floor(plan_param.Boost_margin/0.025));
% SE_1 = strel('disk',floor(plan_param.Antiboost_margin/0.025));


%IMRT_4 margins
plan_param.Boost_margin = 1.2;
disp('setting Boost margin to 1.2')
plan_param.Antiboost_margin = 0.6;
disp('setting AntiBoost margin to 0.6')

BMargin = (plan_param.Boost_margin)/(1.26);
SE = strel('disk',floor(BMargin/0.025));
AMargin = ((plan_param.Antiboost_margin)*1.26);
SE_1 = strel('disk',floor(AMargin/0.025));


%SE_2 = strel('disk',floor((0.3)/(0.025*2))); %Just dilate the tumor mask by 0.3mm to account for printer shrinking.

parfor ii =  1:length(Bev_masks)
        Bev_masks{ii}.Dilated_boost_map = imdilate(Bev_masks{ii}.Boost_map,SE);
        Bev_masks{ii}.Subtract_Dilated_boost_map = imdilate(Bev_masks{ii}.Boost_map,SE_1);
        %Bev_masks{ii}.Double_Dilated_boost_map = imdilate(Bev_masks{ii}.Dilated_boost_map,SE);
        Bev_masks{ii}.Margin_dilated_boost = plan_param.Boost_margin ;        
end

% %This bit takes alot of time and memory. Should only be done as a
% %post_processing step.
% if handles.Construct_coverage_map.Value ==1
% %get coverage mask for the boost
%  parfor ii = 1:length(Bev_masks)
%  [ Bev_masks{ii}.Coverage_mask_Boost ] = mask_bev_quant(Tumor_CT, Hypoxia_CT , Bev_masks{ii}.Dilated_boost_map, Bev_masks{ii}.Angle);  
%  end
%  
% %get coverage mask for the AntiBoost
%  parfor ii = 1:length(Bev_masks)
%  [ Bev_masks{ii}.Coverage_mask_AntiBoost ] = mask_bev_quant(Tumor_CT, Hypoxia_CT , Bev_masks{ii}.Dilated_Anti_boost_map, Bev_masks{ii}.Angle);  
%  end
% end
%  

 disp('Saving plug production dataset')       
%Save all informaton for replication of this plug production in a workspace file 
CT_Frame_data = struct('Tumor_CT', Tumor_CT ,'Hypoxia_CT', Hypoxia_CT ,'EPRI_CT',EPRI_CT );
  save(sprintf('%s',plan_param.Experiment_path,'Plug_production_dataset'),'Tumor_CT','Hypoxia_CT', 'CT', 'Bev_masks','WL_shift','plan_param' ,'CT_Frame_data','-v7.3')
        clear Tumor_CT Hypoxia_CT CT EPRI_CT CT_Frame_data             

    
%This switch handles the boost vs antiboost part.
switch plan_param.Boost_or_antiboost
    case 'Boost'
        

disp('Creating Boost Plug files')
 parfor ii = 1:length(Bev_masks)     
     stl_file = sprintf('%s', plan_param.Experiment_path, plan_param.Experiment_name ,'_Boost','_',num2str(ii),'_',num2str(Bev_masks{ii}.Angle),'.stl');
     scad_file = sprintf('%s', plan_param.Experiment_path, plan_param.Experiment_name ,'_Boost','_',num2str(ii),'_',num2str(Bev_masks{ii}.Angle),'.scad');
     Bev_masks{ii}.stl_filename = stl_file;
     Bev_masks{ii}.scad_filename = scad_file;   
        
        Target = Bev_masks{ii}.Dilated_boost_map;
        %imagesc(Target_dil)
        [Bound_regions,L] = bwboundaries(Target,'noholes');
        
        %imshow(BW)
        %hold on
        %This part was BE's innovation. Just using a geometric
        %approximation rather than exporting the whole image into openscad
        %is about 1000 times faster to render.
        Poly_vector = {}
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
%     Do_UV_shift = get(handles.UVcheckbox1,'Value')
%     if Do_UV_shift
        
%     UV_table_path = 'Z:\CenterProjects\IMRT_3_development\UV_plug_shift_table\UV_plug_shift_table';
%     load(UV_table_path)
    UVShift = [WL_shift{ii}.Plug_X WL_shift{ii}.Plug_Y];
    %UVShift = [5 0] %in mm
%     else
%         UVShift = [0 0]; 
%     end
    
    %Switch handles the difference between the different sizes of plug
    %holders. User dictates and I have cailbrated the settings so they
    %should fit in the plug holder. Subject to print quality.
     switch plan_param.Plug_size
        case '16mm'
      disp('Writing SCAD files and rendering plugs for Boost')
        epr_Contour2OpenSCAD_Plugs('Plug_cut', ...
          scad_file,...
          Poly_vector,...
          Scale_factor, UVShift);
        
        case '18mm'
      disp('Writing SCAD files and rendering plugs for Boost') 
        epr_Contour2OpenSCAD_Plugs_18_mm('Plug_cut', ... 
          scad_file,...
          Poly_vector,...
          Scale_factor, UVShift);
        case '20mm'
      disp('Writing SCAD files and rendering plugs for Boost') 
        epr_Contour2OpenSCAD_Plugs_20_mm('Plug_cut', ... 
          scad_file,...
          Poly_vector,...
          Scale_factor, UVShift);
      
         case '22mm'
       disp('Writing SCAD files and rendering plugs for Boost') 
         epr_Contour2OpenSCAD_Plugs_22_mm('Plug_cut', ... 
          scad_file,...
          Poly_vector,...
          Scale_factor, UVShift);
        
        
        end

      %Path to an openscad so we don't have version control issues.
      Openscad_command_path = 'Z:\CenterHardware\3Dprint\OpenSCAD';
      Openscad_command_str = 'openscad';
      
      %Render plugs
      cd(Openscad_command_path) %Cd to the sharedrive openscad location.       
      Command_string = sprintf('%s',Openscad_command_str,' -o ',Bev_masks{ii}.stl_filename,' ', Bev_masks{ii}.scad_filename);      
      [status , result] = dos(Command_string)
      
 end
         delete(gcp)
 
 
%       %Last step is to output an INI file to the outpath for loading into the pilot
%       %software.
% 	  presciption = 13;
%       if handles.Manual_thresh.Value ==1
%       [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_mask_inputs_manual(CT, Tumor_CT, Hypoxia_CT, Bev_masks, angles, presciption);
%       else          
%        [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_mask_inputs(CT, Tumor_CT, Hypoxia_CT, Bev_masks, angles, presciption);
%       end
%       Beam_plan_INI_write_V3(handles.Out_path, handles.Experiment_tag ,Gantry_angles,presciption/length(Gantry_angles), beamtimes);

      

 
%Refer to comments from boost case
    case 'AntiBoost'
        
    disp('Create the Antiboost bevs by dilating the boost bevs.')
        
parfor ii = 1:length(Bev_masks)
    
        %Represents an erosion by 0.1mm or 4 smallpix
        %SE = ones(floor((Margin/0.025)),floor((Margin/0.025)))
        %Target = imdilate(Target,SE);

        [ Bev_masks{ii} ] = Find_antiboost_2D_mask_4( Bev_masks{ii});
        
        %Dilate the antiboost mask. Make sure to resubtract the boost mask      
%         Bev_masks{ii}.Dilated_Anti_boost_map = imdilate(Bev_masks{ii}.Antiboost_map,SE_2);
        %Bev_masks{ii}.Dilated_Anti_boost_map((Bev_masks{ii}.Subtract_Dilated_boost_map==1))=0;
        %Bev_masks{ii}.Dilated_Anti_boost_map(find(imdilate(Bev_masks{ii}.Boost_map,SE_1)==1))=0;
        %Figure the support mask to keep the dilated antiboost map from
        %being unprintable. 
% In case we ever want to look at each bev        
%          figure(1); imagesc(Bev_masks{ii}.Boost_map);
%          figure(2); imagesc(Bev_masks{ii}.Dilated_boost_map);
%          figure(3); imagesc(Bev_masks{ii}.Antiboost_map);
%          figure(4); imagesc(Bev_masks{ii}.Dilated_Anti_boost_map);
%          pause(1.5)    
end


        
 disp('Creating Anti Boost Plug files')
 for ii = 1:length(Bev_masks)     
     stl_file = sprintf('%s',plan_param.Experiment_path, plan_param.Experiment_name  ,'_Anti_Boost','_',num2str(ii),'_',num2str(Bev_masks{ii}.Angle),'.stl');
     scad_file = sprintf('%s',plan_param.Experiment_path, plan_param.Experiment_name ,'_Anti_Boost','_',num2str(ii),'_',num2str(Bev_masks{ii}.Angle),'.scad');
     Bev_masks{ii}.stl_filename = stl_file;
     Bev_masks{ii}.scad_filename = scad_file;   
        
        Target = Bev_masks{ii}.Antiboost_map;
        %Support = Bev_masks{ii}.Support_mask;

        %imagesc(Target_dil)

        [Bound_regions,L] = bwboundaries(Target,'holes');
        %[Bound_regions_support,L] = bwboundaries(Support,'holes');
        
        %imshow(BW)
        %hold on
         Poly_vector = {} %Convert to centered rather than vectored input
        for k = 1:length(Bound_regions)
           boundary = Bound_regions{k};
           %plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
        
           Poly_vector{k}(:,1) =  boundary(:,1) - size(Target, 1)/2;
           Poly_vector{k}(:,2) =  boundary(:,2) - size(Target, 2)/2;
        end
        
        %investigate the Poly_vector of the mask. Identify any outlines
        %which are internal. These need to be classified differenly so they are added to the plug instead of subtracted.
        %MM 8/30/2016 commented this code out. This way is flawed in that
        %two structural elements can cause each other to be assumed to be
        %outline elements. Rewritten below: 
%         Poly_vectors_struct = {};
%         for k = 1:length(Bound_regions)
%         Outline = zeros(size(Target));
%         idx = sub2ind(size(Target),Bound_regions{k}(:,1),Bound_regions{k}(:,2));
%         Outline(idx) = 1;
%         Outline_filled = imfill(Outline,'holes');
%         if numel(find(Target(find(Outline_filled-Outline))))>1;
%         disp('outside border')
%         else
%         disp('inside border adding to structure')
%         Poly_vectors_struct{end+1} = Bound_regions{k};
%         end
%         end

    %investigate the Poly_vector of the mask. Identify any outlines
    %use a priori statment: There is only one outline, and it is the
    %largest area when filled.    
       
        Outline_area_max = 0;
        for k = 1:length(Bound_regions)
        Outline = zeros(size(Target));
        idx = sub2ind(size(Target),Bound_regions{k}(:,1),Bound_regions{k}(:,2));
        Outline(idx) = 1;
        Outline_filled = imfill(Outline,'holes');
        if numel(find(Outline_filled)) > Outline_area_max
        Outline_idx = k; Outline_area_max  = numel(find(Outline_filled)); 
        end       
        end
%         Poly_vectors_struct = Bound_regions{k};
%         Poly_vectors_struct{Outline_idx} = [];
        
        Poly_vectors_struct = Bound_regions;
        Poly_vectors_struct{Outline_idx} = [];
        
        %figure; imagesc(Target); hold on; for vect = 1:length(Poly_vectors_struct); plot(Poly_vectors_struct{vect}(:,2),Poly_vectors_struct{vect}(:,1), 'm');end
        
        Poly_vectors_structure ={}; %Convert to centered rather than vectored input
        if length(Poly_vectors_struct)> 0
        for k = 1:length(Poly_vectors_struct)
            if length(Poly_vectors_struct{k}) > 1
           boundary = Poly_vectors_struct{k};
           %plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)        
           Poly_vectors_structure{k}(:,1) =  boundary(:,1) - size(Target, 1)/2;
           Poly_vectors_structure{k}(:,2) =  boundary(:,2) - size(Target, 2)/2;
            else
                Poly_vectors_structure{k} = [0 0];
            end
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
%     Do_UV_shift = get(handles.UVcheckbox1,'Value')
%     if Do_UV_shift
%         
%     UV_table_path = 'Z:\CenterProjects\IMRT_3_development\UV_plug_shift_table\UV_plug_shift_table';
%     load(UV_table_path)
    UVShift = [WL_shift{ii}.Plug_X WL_shift{ii}.Plug_Y]; 
    %UVShift = [5 0] %in mm
%     else
%         UVShift = [0 0]; 
%     end
    disp('Writing SCAD files and rendering plugs for anti_boost')
%         epr_Contour2OpenSCAD_Plugs_Plus_Support('Plug_cut','Plug_structures', ...
%           scad_file,...
%           Poly_vector,...
%           Poly_vectors_structure,...
%           Poly_vector_support,...
%           Scale_factor, UVShift);

     switch plan_param.Plug_size
        case '16mm'
      epr_Contour2OpenSCAD_Plugs_1_full_layer_plus_structure('Plug_cut','Plug_structures',...
        scad_file,...
        Poly_vector,Poly_vectors_structure,...
        Scale_factor, UVShift);
        
        case '18mm'
      epr_Contour2OpenSCAD_Plugs_1_full_layer_plus_structure_18_mm('Plug_cut','Plug_structures',...
        scad_file,...
        Poly_vector,Poly_vectors_structure,...
        Scale_factor, UVShift);
         case '20mm'
        epr_Contour2OpenSCAD_Plugs_1_full_layer_plus_structure_20_mm('Plug_cut','Plug_structures',...
        scad_file,...
        Poly_vector,Poly_vectors_structure,...
        Scale_factor, UVShift);
             
        
         case '22mm'
        epr_Contour2OpenSCAD_Plugs_1_full_layer_plus_structure_22_mm('Plug_cut','Plug_structures',...
        scad_file,...
        Poly_vector,Poly_vectors_structure,...
        Scale_factor, UVShift);
             
        
        end


      %epr_Contour2OpenSCAD_Plugs_Plus_Support(mod_name,structure_mod_name, fname, vectors,vectors_structure,vectors_support, scale, UVShift)
      
      Openscad_command_path = 'Z:\CenterHardware\3Dprint\OpenSCAD';
      Openscad_command_str = 'openscad';
      
      %Render plugs
      cd(Openscad_command_path) %Cd to the sharedrive openscad location.       
      Command_string = sprintf('%s',Openscad_command_str,' -o ',Bev_masks{ii}.stl_filename,' ', Bev_masks{ii}.scad_filename);      
      [status , result] = dos(Command_string)
 end
 
  delete(gcp)
 


end
 
 disp(sprintf('%s','Plug production finished. Files are at ', plan_param.Experiment_path ))
 toc(Time_all)

end
