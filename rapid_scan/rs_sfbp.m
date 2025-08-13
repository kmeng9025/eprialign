% RS_SFBP  processing script for sinusoidal Rapid Scan image
% FinalImage = RS_SFBP(file_name, file_suffix, output_path, fields_struct);
% file_name     - [string] path/name/extension of experimental file
% file_suffix   - [string] suffix to be added to the output name
% output_path   - [string] output directory
% fields_struct - [structure] of processing parameters
%     [].fbp - [structure] of projection parameters
%         [].. take_param, nPolar, nAz, nSpec, imtype, MaxGradient, angle_sampling
%         CoordPole, baseline, zero_gradient, bl_n, split_field, Q
%     [].ppr - [structure] of data processing parameters
%         [].. Fconv, baseline_harmonics, scan_phase_algorithm,
%         field_scan_phase, phase_algorithm, data_phase
%     [].rec - [structure] of image reconstruction parameters
%         [].. Sub_points, Size, DoublePoints, InterpFactor, Filter,
%         Interpolation, FilterCutOff, CodeFlag, SubSampling
%         For further details see IRADON_D2D_MSTAGE.
%     [].fit - [structure] of fitting parameters
%     [].clb - [structure] of calibration parameters
%     [].prc - [structure] of script controlling parameters
%         [].process_method - [string (rs_sfbp)]
%         [].recon_data     - Reconstruct image ? [string, (yes) | no] 
%         [].save_data      - Save data ? [string, (yes) | no] 
%         [].fit_data       - Fit data ? [string, yes | (no)] 
% See also IRADON_D2D_MSTAGE, RS_SRECONSTRUCT, RS_SDECONVOLVE, RS_SSCAN_PHASE.

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, 2013
% Contact: epri.uchicago.edu

function FinalImage = rs_sfbp(file_name, file_suffix, output_path, fields)

is_fit_data = strcmp(safeget(fields.prc, 'fit_data','yes'),'yes');
is_recon_data = strcmp(safeget(fields.prc, 'recon_data','yes'),'yes');
is_save_data = strcmp(safeget(fields.prc, 'save_data','yes'),'yes');
is_export_projections =strcmp(safeget(fields.prc, 'export_prj', 'no'), 'raw');

fnames = epri_filename(file_name, file_suffix, output_path);

load_opt.Modality = 'RSFBP';
load_opt.FBP = fields.fbp;

% Load data
[out,FinalImage] = epri_load_for_processing(file_name, load_opt);

is_projection_loaded = false;
if isempty(out.mat), 
  is_projection_loaded = true; 
  is_export_projections = false;
  mat = FinalImage;
end

FBP = FinalImage.raw_info.data.FBP; % FBP structure as loaded from file

% Reconstruction parameters
rec_info = fields;
rec_info.fbp = FBP; % change FBP structure to the one used for data loading
rec_info.fft.FOV = fields.rec.size*FBP.MaxGradient*2.8025; % Field of view of data
rec_info.clb.ampHH = 1;  % hardware signal calibration


% -------------------------------------------------------------------------
% -------------- R E C O N S T R U C T I O N ----------------------------
% -------------------------------------------------------------------------
if is_recon_data
  switch upper(rec_info.rec.CodeFlag)  
      case {'MATLAB'}
          [FinalImage.Raw, FinalImage.rec_info, dsc] = rs_sreconstruct(out.mat, FinalImage.raw_info, rec_info);
          FinalImage.Size = FinalImage.rec_info.rec.size;
      case 'MARK4DV1'
          [FinalImage.Raw, FinalImage.rec_info, dsc] = iradon_d2d_4D_direct(out.mat, FinalImage.raw_info, rec_info);
          FinalImage.Size = FinalImage.rec_info.rec.size;          
  end
  
  % remove projection fields
  if is_projection_loaded
    FinalImage = rmfield(FinalImage, 'G');
    FinalImage = rmfield(FinalImage, 'P');
    FinalImage = rmfield(FinalImage, 'B');
  end
  
  % Image postprocessing
  fields_img = safeget(fields, 'img', []);
  switch safeget(fields_img, 'reg_method', 'none')
    case 'TV-L2'
      par_sigma = safeget(fields_img, 'sigma', 0.001);
      par_tau = safeget(fields_img, 'tau', 0.125);
      par_lambda0 = safeget(fields_img, 'lambda0', 1);
      for ii=1:size(FinalImage.Raw,4)
        FinalImage.Raw(:,:,:,ii) = TV_noise_reduce_3D(FinalImage.Raw(:,:,:,ii), ...
          par_sigma, par_tau, par_lambda0);
      end
  end
  
  if is_export_projections
    epri_create_directory(fnames.path);
    s1 = dsc.raw;
    s1.file_type    = 'Projections_v1.0';
    save(fnames.prj_file,'-struct','s1');
    fprintf('File %s is saved.\n', fnames.prj_file);
  end
  
  FinalImage.raw_info.Boffset = safeget(fields.ppr, 'data_offset', 0) + ...
    -safeget(fields.ppr, 'field_reference', 0);
  
  % Save raw image
  if is_save_data
    epri_create_directory(fnames.path);
    s.file_type    = 'Image_v1.1';
    s.raw_info     = FinalImage.raw_info;
%     s.traces       = single(yyy);
%     s.projections  = single(y);
    s.mat_recFXD   = single(FinalImage.Raw);
    s.rec_info = FinalImage.rec_info;
    s.pO2_info = fields.clb;
    save(fnames.raw_file,'-struct','s');
    fprintf('File %s is saved.\n', fnames.raw_file);
  end
  FinalImage.ImageName = fnames.raw_file;
end

% -------------------------------------------------------------------------
% -------------- D A T A   F I T I N G  -----------------------------------
% -------------------------------------------------------------------------
if ndims(FinalImage.Raw) ~=4, is_fit_data = false; end
if is_fit_data
  FinalImage.raw_info.deltaH = FinalImage.rec_info.rec.deltaH;
  
  
  [FinalImage.fit_data, FinalImage.mat_fit_info] = rs_spectral_fit(FinalImage.Raw, FinalImage.raw_info, fields.fit);
  
  if is_save_data
    epri_create_directory(fnames.path);
    s1.file_type    = 'FitImage_v1.1';
    s1.source_image = fnames.raw_file;
    s1.p_image = fnames.p_file;
    s1.raw_info     = FinalImage.raw_info;
    s1.fit_data     = FinalImage.fit_data;
    s1.rec_info     = FinalImage.rec_info;
    s1.pO2_info     = fields.clb; %#ok<STRNU>
    if strcmp('big_one', 'small_one')
      save(fnames.p_file,'-struct','s1');
    else
      save(fnames.p_file,'-v7.3','-struct','s1');
    end
    fprintf('File %s is saved.\n', fnames.p_file);
  end
  
  % Load fit results for display
  Q_correction  =1;
  [FinalImage.Amp, FinalImage.LW, FinalImage.Mask, FinalImage.Xover] = LoadFitPars(FinalImage.fit_data, {'Amp','LLW','Mask', 'XOVER'});
  FinalImage.pO2 = epr_LLW_PO2(FinalImage.LW*1E3, FinalImage.Amp, FinalImage.Mask, fields.clb);
%   Q_correction = sqrt(FinalImage.pO2_info.Qcb/FinalImage.pO2_info.Q);
  FinalImage.pO2_info.ampHH = FinalImage.raw_info.ampHH;
  FinalImage.pO2_info.ampSS = FinalImage.rec_info.ampSS;
  FinalImage.Amp = FinalImage.Amp*FinalImage.pO2_info.ampHH * FinalImage.pO2_info.ampSS*Q_correction/ fields.clb.amp1mM;
  FinalImage.LW = FinalImage.LW * 1e3; % [mG]

%   [FinalImage.Amp, FinalImage.T2, FinalImage.Mask, FinalImage.Error, eR2] = LoadFitPars(FinalImage.fit_data, {'Amp','T2','Mask','Error', 'Error_R2'});
%   FinalImage.pO2 = epr_T2_PO2(FinalImage.T2, FinalImage.Amp, FinalImage.Mask, fields.clb);
%   Torr_per_mGauss = safeget(fields.clb, 'Torr_per_mGauss', 1.84);
%   FinalImage.Error_O2 = eR2/pi/2/2.8*1000*Torr_per_mGauss;
%   %       Q_correction = sqrt(s1.mat_pO2_info.Qcb/s1.mat_pO2_info.Q);
%   Q_correction = 1;
%   FinalImage.Amp = FinalImage.Amp * Q_correction/ fields.clb.amp1mM;
else
  FinalImage.fit_data = [];
end