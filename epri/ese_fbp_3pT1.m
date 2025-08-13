% FinalImage = ese_fbp(file_name, file_suffix, output_path, fields)
%   file_name:     full name of the file
%   file_suffix:   string to be added before extension
%   output_path:   the path where to store file, name will be preserved,
%                  suffix will be added
%   fields:        structure with fields td, fft, prc, clb etc
% FinalImage = ese_fbp({file_names}, file_suffix, output_path, fields)
%   {file_names}"  a cell array of filenames to be summed before
%                  processing
% FinalImage = ese_fbp(Loaded_Image, file_suffix, output_path, fields)
%   Loaded_Image: a struct with raw_info, mat and mat_bl fields

function FinalImage = ese_fbp_3pT1(file_name, file_suffix, output_path, fields)

is_fit_data = strcmp(safeget(fields.prc, 'fit_data','yes'),'yes');
is_recon_data = strcmp(safeget(fields.prc, 'recon_data','yes'),'yes');
is_export_proj = strcmp(safeget(fields.prc, 'export_prj','yes'),'yes');
is_export_raw = strcmp(safeget(fields.prc, 'export_raw','yes'),'yes');
is_save_data = strcmp(safeget(fields.prc, 'save_data','yes'),'yes');

fnames = epri_filename(file_name, file_suffix, output_path);

load_opt.Modality = 'PULSEFBP';
load_opt.Sequence  = '3pT1';
load_opt.FBP = fields.fbp;

% Load data
[out,FinalImage] = epri_load_for_processing(file_name, load_opt);
FBP = FinalImage.raw_info.data.FBP; % FBP structure as loaded from file

% Reconstruction parameters
rec_info = fields;
rec_info.fbp = FBP; % change FBP structure to the one used for data loading
rec_info.fbp.projection_order = safeget(fields.fbp, 'projection_order', 'default');
rec_info.fft.FOV = fields.rec.Size*FBP.MaxGradient*2.802; % Field of view of data
rec_info.clb.ampHH = 1;  % hardware signal calibration

if strcmp(safeget(rec_info.td, 'prj_transpose', 'no'), 'yes')
  out.mat = real(out.mat) - 1i*imag(out.mat);
  out.mat_bl = real(out.mat_bl) - 1i*imag(out.mat_bl);
end

if strcmp(safeget(rec_info.td, 'off_res_baseline', 'yes'), 'yes')
  yyy = out.mat - out.mat_bl;
  if isfield(out, 'mat_nav')
    yyyN = out.mat_nav - out.mat_bl_nav;
  else
    yyyN = [];
  end
else
  yyy = out.mat;
  if isfield(out, 'mat_nav')
    yyyN = out.mat_nav;
  else
    yyyN = [];
  end
end

other = [];
if strcmp(safeget(rec_info.fft, 'xshift_mode', 'none'), 'fit_gzero_trace')
  other.zerog = out.zero_g - mean(out.mat_bl, 3);
end
if isfield(out, 'zero_g') && ~isempty(out.zero_g)
  other.zerog = out.zero_g;
end

if is_export_raw
    epri_create_directory(fnames.path);
    clear s
    s.raw = yyy;
    if ~isempty(yyyN)
      s.rawN  = yyyN;
    end
    s.raw_info = FinalImage.raw_info;
    save(fnames.prj_rawfile,'-struct','s');
    fprintf('Raw file %s is saved.\n', fnames.prj_rawfile);
end

% -------------------------------------------------------------------------
% -------------- R E C O N S T R U C T I O N ----------------------------
% -------------------------------------------------------------------------
if is_recon_data
  % k-space definition in [cm-1]
  rec_info.td.t2k  = fields.fbp.MaxGradient*2.8025e6/2;
  rec_info.td.kmax = 1/fields.rec.Size*fields.rec.Sub_points/2; % -kmax to kmax
  % process projections
  [x, y, FinalImage.rec_info, dsc] = epri_ese_preprocess(yyy, FinalImage.raw_info, rec_info, other);
  
  % process navigators
  if ~isempty(yyyN)
    rec_info1 = FinalImage.rec_info;
    rec_info1.td.echo_select_algorithm = 'manual';
    rec_info1.td.echo_idx = rec_info1.td.max_idx;
    rec_info1.td.phase_algorithm = 'manual';
    rec_info1.td.phase_zero_order = 0;
    rec_info1.fbp.projection_order='as is';
    [~, yN] = epri_ese_preprocess(yyyN, FinalImage.raw_info, rec_info1);
  end
  
  % reconstruct projections
  if isfield(rec_info.rec, 'recon_echos') && ~isempty(rec_info.rec.recon_echos)
    list_of_echoes = rec_info.rec.recon_echos;
%     if isfield(rec_info.td, 'use_echos') && ~isempty(rec_info.td.use_echos)
%       idx = zeros(length(FinalImage.raw_info.T1), 1);
%       idx(rec_info.td.use_echos) = 1;
%       rec_info.td.use_echos = find(idx(list_of_echoes));
%     end
  else
    list_of_echoes = 1:size(out.mat, 2);
  end
  
  if is_export_proj
    epri_create_directory(fnames.path);
    clear s
    s.P = y;
    if ~isempty(yyyN)
      s.Pnav  = yN;
      s.Gnav    = out.nav_G;
      s.Gexp_nav = out.nav_Gexp;
    end
    s.B = x;
    s.G    = out.G;
    s.Gexp = out.Gexp;
    s.knorm = s.G / FBP.MaxGradient;
    s.raw_info = FinalImage.raw_info;
    if isfield(dsc, 'zerog')
      s.zerog = dsc.zerog;
    end
    save(fnames.prj_file,'-struct','s');
    fprintf('File %s is saved.\n', fnames.prj_file);
  end
  
  % rearrange projections
  [y, out] = epri_msps_decrypt(y, out, FinalImage.rec_info.fbp);
  
  [FinalImage.Raw, FinalImage.rec_info, dsc] = epri_reconstruct(y(:,list_of_echoes,:,:), FinalImage.raw_info, rec_info);
  
  FinalImage.Size = FinalImage.rec_info.rec.Size;

  % Image postprocessing
  fields_img = safeget(fields, 'img', []);
  mirror_image = safeget(fields_img, 'mirror_image', [0,0,0]);
  if mirror_image(1), FinalImage.Raw = flip(FinalImage.Raw, 1); end
  if mirror_image(2), FinalImage.Raw = flip(FinalImage.Raw, 2); end
  if mirror_image(3), FinalImage.Raw = flip(FinalImage.Raw, 3); end
  
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
  
  % Save raw image
  if is_save_data
    epri_create_directory(fnames.path);
    s.file_type    = 'Image_v1.1';
    s.raw_info     = FinalImage.raw_info;
    s.traces       = single(yyy);
    s.projections  = single(y);
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
  selection = 1:size(FinalImage.Raw, 4);
  [FinalImage.fit_data] = epri_decay_fit(FinalImage.Raw, FinalImage.raw_info.T1(selection)'*1E6, fields.fit);
  
  if is_save_data
    epri_create_directory(fnames.path);
    s1.file_type    = 'FitImage_v1.1';
    s1.source_image = fnames.raw_file;
    s1.p_image = fnames.p_file;
    s1.raw_info     = FinalImage.raw_info;
    s1.fit_data     = FinalImage.fit_data;
    s1.rec_info     = FinalImage.rec_info;
    s1.pO2_info     = fields.clb;
    save(fnames.p_file,'-struct','s1');
    fprintf('File %s is saved.\n', fnames.p_file);
  end
  
  % Load fit results for display
  [FinalImage.Amp, FinalImage.T2, FinalImage.Mask, FinalImage.Error, eR2] = LoadFitPars(FinalImage.fit_data, {'Amp','T2','Mask','Error', 'Error_R2'});
  FinalImage.pO2 = epr_T2_PO2(FinalImage.T2, FinalImage.Amp, FinalImage.Mask, fields.clb);
  Torr_per_mGauss = safeget(fields.clb, 'Torr_per_mGauss', 1.84);
  FinalImage.Error_O2 = eR2/pi/2/2.8*1000*Torr_per_mGauss;
  %       Q_correction = sqrt(s1.mat_pO2_info.Qcb/s1.mat_pO2_info.Q);
  Q_correction = 1;
  FinalImage.Amp = FinalImage.Amp * Q_correction/ fields.clb.amp1mM;
else
  FinalImage.fit_data = [];
end

% --------------------------------------------------------------------
function s = copy_fields(s, s1)

fname = fieldnames(s1);
for ii=1:length(fname), s.(fname{ii}) = s1.(fname{ii}); end

