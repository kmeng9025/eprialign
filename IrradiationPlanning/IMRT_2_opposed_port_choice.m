function [ Port_Gantry_Angle , Bev_size ] = IMRT_2_opposed_port_choice( CT , EPRI, Tumor, plan_param )

cut_off= 0.5;


% Time_all= tic %time all the time consuming stuff

%Transform the MRI Tumor mask into the CT
disp('tranforming tumor to CT')
tic
Tumor_CT = reslice_volume(inv(CT.transform),inv(Tumor.transform),zeros(size(CT.data)), double(Tumor.data) , 0, 1) > cut_off;
toc

%Code to produce the plugs goes here:
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
% Table_name = 'Z:\CenterProjects\IMRT_3_development\Plug_Shift_table\Final_plug_shift_table_10_ports.xlsx' ;
% W_L_table = xlsread(Table_name);
% for ii= 1:10
% % WL_shift{ii}.Plug_X = W_L_table(ii,4) + W_L_table(ii,7); 
% % WL_shift{ii}.Plug_Y = W_L_table(ii,5) + W_L_table(ii,8);
% Gantry_angle(ii) = W_L_table(ii,1);
% Plug_X(ii) = W_L_table(ii,3);
% Plug_Y(ii) = W_L_table(ii,4);
% end
% WL_shift = struct('Gantry_angle',Gantry_angle,'Plug_X',Plug_X,'Plug_Y',Plug_Y);
 
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

for ii = 1:length(Bev_masks)
   Bev_size(ii) =  Bev_masks{ii}.Boost_bev_volume;
   
end
Port_Gantry_Angle = Bev_masks{find(Bev_size == min(Bev_size))}.Angle; 



end