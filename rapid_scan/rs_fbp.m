% FinalImage = rs_fbp(file_name, file_suffix, output_path, fields) 
% Rapid scan image reconstruction
function FinalImage = rs_fbp(file_name, file_suffix, output_path, fields)

is_fit_data = strcmp(safeget(fields.prc, 'fit_data','yes'),'yes');
is_recon_data = strcmp(safeget(fields.prc, 'recon_data','yes'),'yes');

% select file names
if iscell(file_name), [fp, fn]=fileparts(file_name{1});
else [fp, fn]=fileparts(file_name);
end
if isempty(output_path)
  savefile = fullfile(fp, [fn, file_suffix,'.mat']);
  psavefile = fullfile(fp, ['p',fn, file_suffix,'.mat']);
else
  savefile = fullfile(output_path, [fn, file_suffix, '.mat']);
  psavefile = fullfile(output_path, ['p', fn, file_suffix, '.mat']);
end

% load data from file (use epr_ReadPulseImageFile)
[out,FinalImage] = epr_load_for_processing(file_name, fields.fbp);
if isempty(FinalImage), return; end

% Prepare output structure
FinalImage.rec_info.ppr = fields.ppr;
FinalImage.rec_info.rec = fields.rec;
FinalImage.rec_info.prc = fields.prc;
FinalImage.rec_info.rec.Enabled = iff(strcmp(safeget(fields.prc, 'recon_data','yes'),'yes'), 'on', 'off');

FinalImage.fit_info = fields.fit;
FinalImage.pO2_info = fields.clb;

%% prepare data for reconstruction

if is_recon_data
  [FinalImage.Raw, FinalImage.rec_info, dsc] = rs_reconstruct(mat, FinalImage.raw_info, FinalImage.rec_info, fields.fit);
  FinalImage.Size = FinalImage.rec_info.rec.Size;
end

if strcmpi(safeget(fields.ppr, 'fft_export_clearence', 'no'), 'yes')
  FinalImage.Clearence = safeget(dsc, 'export_clearence', []);
end

if is_fit_data
  Q_correction  =1;
  FinalImage.raw_info.deltaH = FinalImage.rec_info.rec.deltaH;
  [FinalImage.mat_fit, FinalImage.mat_fit_info] = rs_spectral_fit(FinalImage.Raw, FinalImage.raw_info, FinalImage.fit_info);
  [FinalImage.Amp, FinalImage.LW, FinalImage.Mask] = LoadFitPars(FinalImage.mat_fit, {'Amp','LLW','Mask'});
  FinalImage.pO2 = epr_LLW_PO2(FinalImage.LW*1E3, FinalImage.Amp, FinalImage.Mask, fields.clb);
%   Q_correction = sqrt(FinalImage.pO2_info.Qcb/FinalImage.pO2_info.Q);
  FinalImage.pO2_info.ampHH = FinalImage.raw_info.ampHH;
  FinalImage.pO2_info.ampSS = FinalImage.rec_info.ampSS;
  FinalImage.Amp = FinalImage.Amp*FinalImage.pO2_info.ampHH * FinalImage.pO2_info.ampSS*Q_correction/ fields.clb.amp1mM;
else
  FinalImage.mat_fit = [];
  FinalImage.pO2_info.ampHH = FinalImage.raw_info.ampHH;
end

if strcmp(safeget(FinalImage.rec_info.prc, 'save_data','yes'),'yes')
  s.file_type    = 'Image_v1.1';
  s.raw_info     = FinalImage.raw_info;
  s.mat          = single(mat);
  s.mat_bl       = single(mat_bl);
  if strcmp(safeget(FinalImage.rec_info.prc, 'recon_data','yes'),'yes')
    s.mat_recFXD   = single(FinalImage.Raw);
    s.rec_info = FinalImage.rec_info;
    s.pO2_info = FinalImage.pO2_info;
  end
  if is_fit_data
    s1.file_type    = 'FitImage_v1.1';
    s1.raw_image    = file_name;
    s1.source_image = savefile;
    s1.raw_info     = FinalImage.raw_info;
    s1.fit_data      = FinalImage.mat_fit;
    s1.rec_info = FinalImage.rec_info;
    s1.pO2_info  = FinalImage.pO2_info;
    s1.fit_info = FinalImage.fit_info;
  end
  
  td_CreateDirectory(savefile);
  if exist(savefile, 'file')
    save(savefile,'-struct','s','-append');
  else
    save(savefile,'-struct','s');
  end
  if is_fit_data
    if exist(psavefile, 'file')
      save(psavefile,'-struct','s1','-append');
    else
      save(psavefile,'-struct','s1');
    end
  end
  disp(sprintf('    Data are saved to %s.', savefile));
end
