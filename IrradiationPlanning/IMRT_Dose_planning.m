function [  ] = IMRT_Dose_planning( CT , EPRI, Tumor, plan_param )

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



CT_frame = struct('CT',CT,'EPRI_CT',EPRI_CT,'Tumor_CT',Tumor_CT,'Hypoxia_CT',Hypoxia_CT);

[ Planning_Output ] = Generalized_Dose_planning_func( CT_frame , plan_param )

% %Everything below is irrelevant code made obsolete by the function above.
% Leave in for posterity.
% %Flag for boost vs antiboost Obsolete, but just leave it because it might
% %break something.
% mask_out_boost= true;
% 
% delete(gcp('nocreate'))
% %init Parallel computing 
% parpool(feature('numCores'))
% 
% disp('Generate Bevs for the target volume')
% %(hypoxia_CT) 
% [Bev_masks] = maskbev( plan_param.Gantry_angles , Tumor_CT, Hypoxia_CT, mask_out_boost  );
%  
% BMargin = (plan_param.Boost_margin)/(1.26);
% SE = strel('disk',floor(BMargin/0.025));
% AMargin = ((plan_param.Antiboost_margin)*1.26);
% SE_1 = strel('disk',floor(AMargin/0.025));
% 
% disp('Dilate masks')
% parfor ii =  1:length(Bev_masks)
%         Bev_masks{ii}.Dilated_boost_map = imdilate(Bev_masks{ii}.Boost_map,SE);
%         Bev_masks{ii}.Subtract_Dilated_boost_map = imdilate(Bev_masks{ii}.Boost_map,SE_1);
%         %Bev_masks{ii}.Double_Dilated_boost_map = imdilate(Bev_masks{ii}.Dilated_boost_map,SE);
%         Bev_masks{ii}.Margin_dilated_boost = plan_param.Boost_margin ;        
% end
% 
% 
% %SE_2 = strel('disk',floor((0.3)/(0.025*2))); %Just dilate the tumor mask by 0.3mm to account for printer shrinking.
% 
% parfor ii =  1:length(Bev_masks)
%         Bev_masks{ii}.Dilated_boost_map = imdilate(Bev_masks{ii}.Boost_map,SE);
%         Bev_masks{ii}.Subtract_Dilated_boost_map = imdilate(Bev_masks{ii}.Boost_map,SE_1);
%         %Bev_masks{ii}.Double_Dilated_boost_map = imdilate(Bev_masks{ii}.Dilated_boost_map,SE);
%         Bev_masks{ii}.Margin_dilated_boost = plan_param.Boost_margin ;        
% end
%  
% delete(gcp)
%       %output an INI file to the outpath for loading into the pilot
%       %software.
% 	  presciption = plan_param.Boost_dose;
%       switch  plan_param.Find_skin_method
%            case 'Normal threshold Method'
%         [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_mask_inputs(CT.data, Tumor_CT, Hypoxia_CT, Bev_masks, plan_param.Gantry_angles, presciption);
%           case 'Manual Method          '
%         [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_mask_inputs_manual(CT.data, Tumor_CT, Hypoxia_CT, Bev_masks, plan_param.Gantry_angles, presciption);
%           case 'Fit circle Method      '
%          [bevmaps, material_mask, depth_info, beamtimes] = maskbev_depths_func_circle_fit(CT.data, Tumor_CT, Hypoxia_CT, Bev_masks, plan_param.Gantry_angles, presciption)
%     
%           
%           case 'Kmeans Method'
%           [Masks] = Function(CT.data)   
%       end
%       
%   switch  safeget(plan_param,'Boost_or_antiboost','');
%     case 'Boost'
%     case 'AntiBoost'
%       beamtimes = beamtimes * (12/7);
%     otherwise
%       
%   end
%       
%     Beam_plan_INI_write_V3(plan_param.Experiment_path, plan_param.Experiment_name,plan_param.Gantry_angles,presciption/length(plan_param.Gantry_angles), beamtimes);

end

