function [ Planning_Output ] = Generalized_Dose_planning_func( CT_Frame_data , plan_param )

%This function takes inputs of CT, EPRI, Tumor with their Transformations.
% and a planning_param struct. It plans a finite number of discrete beams.

%Plan_param requirements =  plan_param.Gantry_angles,
%plan_param.Boost_margin
% plan_param.presciption , plan_param.Conformal_method ,
% plan_param.Boost_or_Anti_boost, plan_param.Output_ini , plan_param.Find_skin_method

%If you output .ini expect to have a path and experiment name for output.


%There may be some legacy code in this function to attempt to plan the beam
%in the same way as the openscource treatment planning program. Use
%plan_param.Conformal_method = 'legacy' access those outputs.
%MM 12/22/2016

%Example paramter set for testing
% plan_param = struct('Gantry_angles',[90 -90], 'Boost_margin',1.2,'presciption',22.5,'Conformal_method','Whole field treatment APPA','Boost_or_antiboost','Boost','Output_ini',0,'Find_skin_method','Normal threshold Method')

cut_off = 0.5 ;

Time_all= tic %time all the time consuming stuff

%Collect The CT frame data to use in planning the beams.
CT = safeget(CT_Frame_data,'CT',[]);
Tumor_CT = safeget(CT_Frame_data,'Tumor_CT',[]);
EPRI_CT = safeget(CT_Frame_data,'EPRI_CT',[]);
Hypoxia_CT = safeget(CT_Frame_data,'Hypoxia_CT',[]);


switch safeget(plan_param,'Conformal_method','Whole field treatment APPA');
  case {'5 Port Equal Spacing' , '2 Port Opposed Beams'}
    %Flag for boost vs antiboost Obsolete, but just leave it because it might
    %break something.
    mask_out_boost= true;
    delete(gcp('nocreate'))
    %init Parallel computing
    parpool(feature('numCores'))
    disp('Generate Bevs for the target volume')
    %(hypoxia_CT)
    [Bev_masks] = maskbev( plan_param.Gantry_angles , Tumor_CT, Hypoxia_CT, mask_out_boost  );
    BMargin = (plan_param.Boost_margin)/(1.26);
    SE = strel('disk',floor(BMargin/0.025));
    disp('Dilate masks')
    parfor ii =  1:length(Bev_masks)
      Bev_masks{ii}.Dilated_boost_map = imdilate(Bev_masks{ii}.Boost_map,SE);
      Bev_masks{ii}.Subtract_Dilated_boost_map = imdilate(Bev_masks{ii}.Boost_map,SE);
      %Bev_masks{ii}.Double_Dilated_boost_map = imdilate(Bev_masks{ii}.Dilated_boost_map,SE);
      Bev_masks{ii}.Margin_dilated_boost = plan_param.Boost_margin ;
    end
    delete(gcp)
    %output an INI file to the outpath for loading into the pilot
    %software.
    presciption = safeget(plan_param,'presciption',11);
    
    switch  safeget(plan_param,'Find_skin_method','Generalized_beam_plan')
      case 'Generalized_beam_plan'
        [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_general(CT.data, Tumor_CT, Hypoxia_CT, Bev_masks, plan_param.Gantry_angles, presciption, 1 );
        
      case 'Normal threshold Method'
        [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_mask_inputs(CT.data, Tumor_CT, Hypoxia_CT, Bev_masks, plan_param.Gantry_angles, presciption);
      case 'Manual Method          '
        [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_mask_inputs_manual(CT.data, Tumor_CT, Hypoxia_CT, Bev_masks, plan_param.Gantry_angles, presciption);
      case 'Fit circle Method      '
        [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_circle_fit(CT.data, Tumor_CT, Hypoxia_CT, Bev_masks, plan_param.Gantry_angles, presciption)
        
        
      case 'Kmeans Method'
        %Never actually got around to implementing this.
        [Masks] = Function(CT.data)
    end
    
    switch  plan_param.Boost_or_antiboost
      case 'Boost'
      case 'AntiBoost'
        beamtimes = beamtimes * (12/7);
    end
    
  case 'Whole field treatment APPA'
    
    %Create maps of circular fields 3.5 diameters. pixel size
    %is 0.025 mm ^ 2
    plan_param.Gantry_angles = [90 -90];
    Field_size_mm = safeget(plan_param, 'Field_size_mm',35);
    presciption = safeget(plan_param,'prescription',10);
    Center_postion = 1;
    
    d = round((Field_size_mm/0.025)/1.32); %Standard Whole field radius in Bev sized pixels.
    %Divide by the mag factor because the planning assumes you are talking about exit plane
    Bev_masks={};
    for ii =1:length(plan_param.Gantry_angles)
      [ Bev_masks{ii}.Dilated_boost_map ] = epr_create_circular_mask( [d*2 d*2] , d );
    end
    Target = CT.data *0 ; %Empty Target, just holds place because we are using center targeting.
    
    switch  safeget(plan_param,'Find_skin_method','Generalized_beam_plan')
      
      case 'Normal threshold Method'
        CTmu=CT;
        if max(CTmu(:))>100,
          CTmu=(CT+1000)/5000;
        end
        [skinmask,ringmask,pvmask]=findskin(CTmu);
        material_mask = CT*0;
        material_mask(find(ringmask)) =2;
        material_mask(find(pvmask))=3;
        material_mask(find(skinmask))=1;
        
        
      case 'Manual Method          '
        
        Thresholds = FindskinGUI(CT.data)
        skinmask = CT.data*0;
        skinmask(find(CT.data<Thresholds.Skin_high & CT.data>Thresholds.Skin_low))=1;
        % skinmask = imerode(imdilate(skinmask,se),se);
        pvmask = CT.data*0;
        pvmask(find(CT.data<Thresholds.PVS_high & CT.data>Thresholds.PVS_low))=1;
        ringmask = CT.data*0;
        ringmask(find(CT.data<Thresholds.Ring_high & CT.data>Thresholds.Ring_low))=1;
        
        material_mask = CT.data*0;
        material_mask(find(skinmask))=1;
        material_mask(find(pvmask))=3;
        material_mask(find(ringmask)) =2;
    end
    
    
    [bevmaps, depth_info, beamtimes] = maskbev_depths_func_general(CT.data, Target, material_mask, Bev_masks, plan_param.Gantry_angles, presciption, Center_postion );
    
    %       switch  safeget(plan_param,'Find_skin_method','Generalized_beam_plan')
    %           case 'Generalized_beam_plan'
    
    %           case 'Normal threshold Method'
    %         [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_mask_inputs(CT.data, Tumor_CT, Hypoxia_CT, Bev_masks, plan_param.Gantry_angles, presciption);
    %           case 'Manual Method          '
    %         [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_mask_inputs_manual(CT.data, Tumor_CT, Hypoxia_CT, Bev_masks, plan_param.Gantry_angles, presciption);
    %           case 'Fit circle Method      '
    %         [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_circle_fit(CT.data, Tumor_CT, Hypoxia_CT, Bev_masks, plan_param.Gantry_angles, presciption)
    
    
    
  case 'Legacy dose planning using open_source_treatment_planning'
    %Implement without involving the GUI.
    
    
    
end
%output an INI file to the outpath for loading into the pilot
%software.
if safeget(plan_param,'Output_ini',0);
  switch  safeget(plan_param,'Conformal_method','Whole field treatment APPA');
    case 'Whole field treatment APPA'
      Beam_plan_INI_write_V3(plan_param.Experiment_path, sprintf('%s',plan_param.Experiment_name,'_Whole_field'),plan_param.Gantry_angles,presciption/length(plan_param.Gantry_angles), beamtimes);
      disp(sprintf('%s','Wrote .ini file to ',plan_param.Experiment_path, plan_param.Experiment_name));
    otherwise
      Beam_plan_INI_write_V3(plan_param.Experiment_path, plan_param.Experiment_name,plan_param.Gantry_angles,presciption/length(plan_param.Gantry_angles), beamtimes);
      disp(sprintf('%s','Wrote .ini file to ',plan_param.Experiment_path, plan_param.Experiment_name));
  end
end

Planning_Output = struct('Beam_time_vector',beamtimes,'depth_info',depth_info , 'material_mask', material_mask, 'Bev_masks', Bev_masks , 'presciption' , presciption );


end

