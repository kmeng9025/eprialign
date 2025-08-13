% FinalImage = fid_spi(file_name, file_suffix, output_path, fields)
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

function FinalImage = fid_spi_fpt(file_name, file_suffix, output_path, fields)

is_fit_data = strcmp(safeget(fields.prc, 'fit_data','yes'),'yes');
is_recon_data = strcmp(safeget(fields.prc, 'recon_data','yes'),'yes');
is_export_proj = strcmp(safeget(fields.prc, 'export_prj','yes'),'yes');
is_save_data = strcmp(safeget(fields.prc, 'save_data','yes'),'yes');

fnames = epri_filename(file_name, file_suffix, output_path);

load_opt.Modality = 'PULSESPI';
load_opt.SPI = fields.spi;
load_opt.Sequence = 'FID';
[out,FinalImage] = epri_load_for_processing(file_name, load_opt);
SPI = FinalImage.raw_info.data.SPI; % SPI structure as loaded from file

rec_info.td = fields.td;
rec_info.fft = fields.fft;
% rec_info.fft.FOV = fields.rec.Size*fields.fbp.MaxGradient*2.802;
rec_info.rec = fields.spi;
rec_info.rec = copy_fields(rec_info.rec, fields.rec);
rec_info.rec.Enabled = iff(strcmp(safeget(fields.prc, 'recon_data','yes'),'yes'), 'on', 'off');
rec_info.prc = fields.prc;

% Pulse data do not have Hardware calibration
fields.clb.ampHH = 1;

if strcmp(safeget(rec_info.td, 'off_res_baseline', 'yes'), 'yes') && ~isempty(out.mat_bl)
  yyy = out.mat - out.mat_bl;
else
  yyy = out.mat;
end

is_filter = true;
if is_filter
  yyy = smoothdata(yyy, 'sgolay', 15);  
%   MatrixGUI(yyy)
end

if is_recon_data
  
  FinalImage.Raw = [];
  FinalImage.rec_info = rec_info;
  
  gidx = FinalImage.raw_info.gidx(FinalImage.raw_info.gidx ~= 0);
  Gmax = FinalImage.raw_info.MaxGradient;
  
  dt = mean(diff(FinalImage.raw_info.t_ax));
  tm = FinalImage.raw_info.t_ax*1E6 + rec_info.td.acq_window_start*1E-9*1E6; % us
  Size = FinalImage.rec_info.rec.Size; %cm
  Dim  = FinalImage.raw_info.Dim; Dim = Dim(1);
  kcomp = safeget(FinalImage.rec_info.rec, 'kcomp', 1.0);
  dG = 2 * Gmax / Dim;
  SPIdelay = kcomp / (Size * dG *2.802e6) * 1e6; %us
  
  [~, idx] = min((tm-SPIdelay).^2);
  FinalImage.rec_info.Size = Size;
  fprintf('Selected point: %i(%i) (%f us)\n', idx, length(tm), SPIdelay)
  yTPtmp = zeros(FinalImage.raw_info.Dim);
  
  acq_intertrigger = safeget(FinalImage.rec_info.td, 'acq_intertrigger', 1000);
  idx_delay = floor(SPIdelay*1E-6 / dt + 0.5);
  idx_offset = floor(acq_intertrigger*1E-9 / dt + 0.5);
  spi_idx = idx:idx_offset:length(FinalImage.raw_info.t_ax);
  echo_idx = idx-idx_delay+idx_offset*2:idx_offset:length(FinalImage.raw_info.t_ax);
  echo_tp = idx-idx_delay+idx_offset*2:idx_offset:length(FinalImage.raw_info.t_ax);
  
    %   for ii=1:size(yyy, 3)
  %     yyy(:, pnt_n, ii) = datasmooth(yyy(:, pnt_n, ii), 6);
  %   end
  for ii=1:length(spi_idx)
    yTPtmp(gidx) = squeeze(yyy(spi_idx(ii), 1 ,:));
    yTPtmp = yTPtmp * exp(1i*rec_info.td.phase_zero_order*pi/180);
    yTP(:,:,:, ii) = yTPtmp;
  end
  
  meanTrace = mean(yyy(:, 1 ,:), 3)*exp(1i*rec_info.td.phase_zero_order*pi/180);
  figure(1000); clf; plot(tm, real(meanTrace), tm, imag(meanTrace), ...
    tm(spi_idx), real(meanTrace(spi_idx)), 'ko', ...
    tm(echo_tp), real(meanTrace(echo_tp)), 'ro');
  grid on; legend({'Mean Re', 'Mean Im', 'tp', 'echo-tp'})
  
  %     figure(100); clf;
  %     for ii=2913-5:2913+5 % 1:size(mat, 3)
  %       trace = squeeze(yyy(idx-50:end, pnt_n, ii));
  %       plot(tm(idx-50:end), real(trace), tm(idx-50:end), imag(trace));
  %       text(0.1, 0.9, sprintf('%d : %d', pnt_n, ii), 'Units', 'normalized');
  %       axis tight
  %       pause
  %     end
  
  %     d_idx = yTP ~= 0;
  %     bl = d_idx & imerode(d_idx, epr_strel('sphere', 2));
  %     im_bl = mean(yTP(bl(:)));
  %     yTP(d_idx) = yTP(d_idx) - im_bl;
  
  %    right_phase = rec_info.td.phase_zero_order*pi/180*ones(length(use_echos),sz(3));
  
  %   yTP = squeeze(sum(yy(idx-1:idx+1,:,:,:),1));
  %
  %   disp(handles.mat_info.Dim/2/tp/Gmax/2.8)
  % reconstruction
  switch 'as_is'
    case 'as_is'
      for ii=1:length(spi_idx)
        yTPtmp = yTP(:,:,:,ii);
        yTPzero = zeros(FinalImage.raw_info.Dim*3);
        yTPzero(FinalImage.raw_info.Dim(1)+1:2*FinalImage.raw_info.Dim(1),...
          FinalImage.raw_info.Dim(2)+1:2*FinalImage.raw_info.Dim(2),...
          FinalImage.raw_info.Dim(3)+1:2*FinalImage.raw_info.Dim(3)) = yTPtmp;
        FinalImage.Raw(:,:,:,ii) = real(fftshift(ifftn(ifftshift(yTPzero))));
      end
      FinalImage.Size = FinalImage.rec_info.rec.Size;
      %      handles.mat_rec_info
      %     case 'interpolation'
      %       FXD = real(fftshift(ifftn(ifftshift(yTP))));
      %       zf_size = handles.mat_info.Dim/2/tp/Gmax/2.8;
      %
      %       scale = mean(handles.mat_rec_info.SizeObj ./ zf_size);
      %
      %       x = handles.mat_info.Dim(1)/2 + 0.5 + ...
      %         ((1 : handles.mat_info.Dim(1)) - handles.mat_info.Dim(1)/2 - 0.5) * scale;
      %       y = handles.mat_info.Dim(2)/2 + 0.5 + ...
      %         ((1 : handles.mat_info.Dim(2)) - handles.mat_info.Dim(2)/2 - 0.5) * scale;
      %       z = handles.mat_info.Dim(3)/2 + 0.5 + ...
      %         ((1 : handles.mat_info.Dim(3)) - handles.mat_info.Dim(3)/2 - 0.5) * scale;
      %       [xi,yi,zi] = meshgrid(y,x,z);
      %       handles.mat_recFXD(:,:,:,pnt_n)  = interp3(FXD, xi,yi,zi)*scale^3;
      %       handles.mat_rec_info.Dim = handles.mat_info.Dim;
      %     case 'zero_filling'
      %       zf_size = round(smooth_factor * 2*tp*Gmax*handles.mat_rec_info.SizeObj);
      %       yTPzero = zeros(handles.mat_info.Dim*3);
      %       yTPzero(handles.mat_info.Dim(1)+1:2*handles.mat_info.Dim(1),...
      %         handles.mat_info.Dim(2)+1:2*handles.mat_info.Dim(2),...
      %         handles.mat_info.Dim(3)+1:2*handles.mat_info.Dim(3)) = yTP;
      %       % ShowSlice3DFIG(100, real(yTP), [])
      %
      %       %       x = ((1:handles.mat_info.Dim(1)*3)-handles.mat_info.Dim(1)*3/2 - 0.5)/2/tp/Gmax/3;
      %       %       y = ((1:handles.mat_info.Dim(2)*3)-handles.mat_info.Dim(2)*3/2 - 0.5)/2/tp/Gmax/3;
      %       %       z = ((1:handles.mat_info.Dim(3)*3)-handles.mat_info.Dim(3)*3/2 - 0.5)/2/tp/Gmax/3;
      %       handles.mat_recFXD(:,:,:,pnt_n) = real(fftshift(ifftn(ifftshift(yTPzero))));
      %       handles.mat_rec_info.SizeObj = handles.mat_info.Dim/2/tp/Gmax/3;
  end
  
  if strcmpi(safeget(fields.fft, 'fft_export_clearence', 'no'), 'yes')
    FinalImage.Clearence = safeget(dsc, 'export_clearence', []);
  end
  
  % Image postprocessing
  fields_img = safeget(fields, 'img', []);
  mirror_image = safeget(fields_img, 'mirror_image', [0,0,0]);
  if mirror_image(1), FinalImage.Raw = flip(FinalImage.Raw, 1); end
  if mirror_image(2), FinalImage.Raw = flip(FinalImage.Raw, 2); end
  if mirror_image(3), FinalImage.Raw = flip(FinalImage.Raw, 3); end
  
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
elseif exist('s', 'var')
  FinalImage.Raw = s.mat_recFXD;
  FinalImage.rec_info = s.rec_info;
  is_recon_data = 1;
end

FinalImage.ImageName = fnames.raw_file;
if is_fit_data
  [FinalImage.fit_data] = td_recovery_fit(FinalImage.Raw, FinalImage.raw_info.T(:)'*1E6, fields.fit);
  
  [FinalImage.Amp, FinalImage.T2, FinalImage.Mask, FinalImage.Error, FinalImage.Error_T2] = LoadFitPars(FinalImage.fit_data, {'Amp','T2','Mask','Error', 'Error_T2'});
  FinalImage.pO2 = epr_T2_PO2(FinalImage.T2, FinalImage.Amp, FinalImage.Mask, fields.clb);
  %       Q_correction = sqrt(s1.mat_pO2_info.Qcb/s1.mat_pO2_info.Q);
  Q_correction = 1;
  FinalImage.Amp = FinalImage.Amp * Q_correction/ fields.clb.amp1mM;
else
  FinalImage.fit_data = [];
end

if is_fit_data
  s1.file_type    = 'FitImage_v1.1';
  s1.raw_image    = file_name;
  s1.source_image = savefile;
  s1.raw_info     = FinalImage.raw_info;
  s1.fit_data      = FinalImage.fit_data;
  s1.rec_info     = FinalImage.rec_info;
  s1.pO2_info     = fields.clb;
  %     s1.raw_info.pars_out = FinalImage.raw_info.pars_out;
end

if is_fit_data
  save(psavefile,'-struct','s1');
end

% --------------------------------------------------------------------
function s = copy_fields(s, s1)

fname = fieldnames(s1);
for ii=1:length(fname), s.(fname{ii}) = s1.(fname{ii}); end

