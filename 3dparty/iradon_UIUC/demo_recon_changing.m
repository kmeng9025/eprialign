%% Generate equal uniform spatial angle layout of projections

fbp_struct.nAz = 18;
fbp_struct.nPolar = 18;
fbp_struct.nSpec = 1;
fbp_struct.imtype = iradon_GetFBPImageType('XYZ');
fbp_struct.MaxGradient = 3.024;
fbp_struct.deltaL = 5;
% fbp_struct.angle_sampling = 'UNIFORM_ANGULAR_FLIP';
fbp_struct.angle_sampling = 'UNIFORM_SPATIAL_FLIP';
fbp_struct.projection_order = 'msps';

[pars, pars_ext] = iradon_FBPGradTable(fbp_struct);

% assign parameters for projection generation
radon_pars.x = pars_ext.kx;
radon_pars.y = pars_ext.ky;
radon_pars.z = pars_ext.kz;
radon_pars.w = pars_ext.w;

%% Calculate weights
% [areas] = vor_area_3d(radon_pars.x,radon_pars.y,radon_pars.z);
% Rec.Phi   = pars_ext.phi;
% Rec.Theta = pars_ext.theta;
% [p,Pars,Rec] = geo_Weights3D([],[],Rec);
% figure; plot(1:pars.nP, (areas - pars_ext.w) ./ pars_ext.w, 1:pars.nP, (Rec.Wt - pars_ext.w) ./ pars_ext.w);

% radon_pars.w = areas;

%% generate 3D projections

R = 2.2; % Radius of the big sphere
Rs = 0.5; % Radius of the small spheres
a = 1.3;  % offset value compared to the center
nBins = 64;
radon_pars.size = 5;

% ksum field is used to determine the summing coefficient
stat_amp = ones(pars.nP, 1);
sin_amp = sin((1:pars.nP)' / pars.nP * pi);
sin_R    = 1.0+0.25*sin((1:pars.nP)' / pars.nP * 2*pi);
step_R   = ones(pars.nP, 1);
step_R(1:pars.nP/2) = 1/2;

phan{1} = struct('object', 'sphere', 'nBins', nBins, 'r', R, 'offset', [0,0,0], 'ksum', stat_amp, 'R', 1/3); % sphere data
phan{2} = struct('object', 'sphere', 'nBins', nBins, 'r', Rs, 'offset', [0,0,0], 'ksum', 2*stat_amp, 'R', 1/2); % sphere data
phan{3} = struct('object', 'sphere', 'nBins', nBins, 'r', Rs, 'offset', [a,0,0], 'ksum', 2*stat_amp, 'R', 1/2); % sphere data
phan{4} = struct('object', 'sphere', 'nBins', nBins, 'r', Rs, 'offset', [-a,0,0], 'ksum', 2*stat_amp, 'R', 1/2); % sphere data
phan{5} = struct('object', 'sphere', 'nBins', nBins, 'r', Rs, 'offset', [0,a,0], 'ksum', 2*stat_amp, 'R', 1/2); % sphere data
phan{6} = struct('object', 'sphere', 'nBins', nBins, 'r', Rs, 'offset', [0,-a,0], 'ksum', 2*stat_amp, 'R', step_R); % sphere data
phan{7} = struct('object', 'sphere', 'nBins', nBins, 'r', Rs, 'offset', [0,0,a], 'ksum', 2*stat_amp, 'R', 1/2); % sphere data
phan{8} = struct('object', 'sphere', 'nBins', nBins, 'r', Rs, 'offset', [0,0,-a], 'ksum', 2*stat_amp, 'R', 1/2); % sphere data

% create holes in the phantom
for ii=2:8
  phan{8+ii-1} = phan{ii};
  phan{8+ii-1}.ksum = -phan{1}.ksum;
  phan{8+ii-1}.R = phan{1}.R;
end

% generate phantom
tau = [0.25, 0.5, 1, 2, 20];
nIm = length(tau);
P = zeros(phan{1}.nBins, pars.nP, nIm);
for nphantom = 1:length(phan)
  for itau = 1:nIm
   PP = radon_c2d_sphere(phan{nphantom}, radon_pars);
   amp = phan{nphantom}.ksum.*(1-2*exp(-tau(itau).*phan{nphantom}.R));
   P(:,:,itau) = P(:,:,itau) + PP.*repmat(amp', nBins, 1);
  end
end
P = permute(P, [1,3,2]);

%% interpolaton

% Rec.Phi   = pars_ext.phi;
% Rec.Theta = pars_ext.theta;
% [p,Pars,Rec] = interp_tri3D(squeeze(P(:,1,:)),[],Rec);

%% FBP image reconstruction (epri_reconstruct)

FinalImage.raw_info = pars;

fields.fbp = fbp_struct;
fields.rec.Sub_points = 64; 
fields.rec.Size = 5;
fields.rec.Filter = 'ram-lak';
fields.rec.FilterCutOff = 0.75;
fields.rec.Interpolation = 'spline';
fields.rec.InterpFactor = 4; % better 4
% fields.rec.CodeFlag = 'MATLAB';
% fields.rec.CodeFlag = 'SINGLEv1';
fields.rec.CodeFlag = 'SINGLEv2 GPU';

fields.rec.zeropadding = 1; % any number >= 1

fields.rec.projection_index = 1:size(P,3)/2;
% fields.rec.projection_index = size(P,3)/2+1:size(P,3);
% fields.fbp.projection_index = [];

[FinalImage.Raw, FinalImage.rec_info, dsc] = epri_reconstruct(P, FinalImage.raw_info, fields);
FinalImage.raw_info.tau2 = ones(length(tau), 1)*2;
FinalImage.raw_info.data.Modality = 'PULSEFBP';
FinalImage.raw_info.data.Sequence = 'ESEInvRec';
FinalImage.raw_info.T1 = tau;
FinalImage.Size = FinalImage.rec_info.rec.Size;

ibGUI(FinalImage)

%% Fit

fields.fit.fit_mask_threshold = 0.25;
fields.fit.fit_method = 'lookup_general';
% fit_par_inv should be slightly above 1.0, depending on first delay
fields.fit.fit_par_inv = 'linspace(1.0, 1.15, 15)'; 
fields.fit.fit_par_R1 = 'linspace(1/0.8, 1/3.5, 300)';
[FinalImage.fit_data] = epri_recovery_fit(FinalImage.Raw, tau(:)', fields.fit);
[FinalImage.Amp, FinalImage.T1, FinalImage.Mask, FinalImage.Error, eR1] = LoadFitPars(FinalImage.fit_data, {'Amp','T1','Mask','Error', 'Error_R1'});

ibGUI(FinalImage);
%% Sliding window reconstruction
FinalImage.raw_info = pars;

fields.fbp = fbp_struct;
fields.rec.Sub_points = 64; 
fields.rec.Size = 5;
fields.rec.Filter = 'ram-lak';
fields.rec.FilterCutOff = 0.75;
fields.rec.Interpolation = 'spline';
fields.rec.InterpFactor = 4; % better 4
% fields.rec.CodeFlag = 'MATLAB';
% fields.rec.CodeFlag = 'SINGLEv1';
fields.rec.CodeFlag = 'SINGLEv2 GPU';

fields.rec.zeropadding = 1; % any number >= 1

fields.fit.fit_mask_threshold = 0.25;
fields.fit.fit_method = 'lookup_general';
% fit_par_inv should be slightly above 1.0, depending on first delay
fields.fit.fit_par_inv = 'linspace(1.0, 1.15, 15)';
fields.fit.fit_par_R1 = 'linspace(1/0.8, 1/3.5, 300)';

fields.clb.LLW_zero_po2 = 9.4;
fields.clb.mG_per_mM = 0.0;
fields.clb.Torr_per_mGauss = 1.8400;
fields.clb.amp1mM = 1;

window = 100;
for ii = 1:4:(size(P,3)-window)
  fields.rec.projection_index = (1:window) + (ii-1);
  
  [FinalImage.Raw, FinalImage.rec_info, dsc] = epri_reconstruct(P, FinalImage.raw_info, fields);
  FinalImage.raw_info.tau2 = ones(length(tau), 1)*2;
  FinalImage.raw_info.data.Modality = 'PULSEFBP';
  FinalImage.raw_info.data.Sequence = 'ESEInvRec';
  FinalImage.raw_info.T1 = tau;
  FinalImage.Size = FinalImage.rec_info.rec.Size;
  
  [FinalImage.fit_data] = epri_recovery_fit(FinalImage.Raw, tau(:)', fields.fit);
  [FinalImage.Amp, FinalImage.T1, FinalImage.Mask, FinalImage.Error, eR1] = LoadFitPars(FinalImage.fit_data, {'Amp','T1','Mask','Error', 'Error_R1'});
  
  
  data_path = 'g:\Projects\SlidingWindow';
  
  % Save raw data
  epri_create_directory(data_path);
  s.file_type    = 'Image_v1.1';
  s.raw_info     = FinalImage.raw_info;
  s.mat_recFXD   = single(FinalImage.Raw);
  s.rec_info = FinalImage.rec_info;
  s.pO2_info = fields.clb;
  data_name = ['slide', num2str(ii), '.mat'];
  save(fullfile(data_path, data_name),'-struct','s');
  fprintf('File %s is saved.\n', fullfile(data_path, data_name));
  
  % save fit data
  s1.file_type    = 'FitImage_v1.1';
  s1.source_image = data_name;
  s1.raw_info     = FinalImage.raw_info;
  s1.fit_data     = FinalImage.fit_data;
  s1.rec_info     = FinalImage.rec_info;
  s1.pO2_info     = fields.clb;
  data_name = ['pslide', num2str(ii), '.mat'];
  save(fullfile(data_path, data_name),'-struct','s1');
  fprintf('File %s is saved.\n', fullfile(data_path, data_name));
end
