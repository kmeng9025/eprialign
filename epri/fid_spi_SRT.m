% FinalImage = fid_spi_SRT(file_name, file_suffix, output_path, fields)
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

function FinalImage = fid_spi_SRT(file_name, file_suffix, output_path, fields)

is_fit_data = strcmp(safeget(fields.prc, 'fit_data','yes'),'yes');
is_recon_data = strcmp(safeget(fields.prc, 'recon_data','yes'),'yes');

load_opt.SPI = fields.spi;
rec_info.td = fields.td;
rec_info.fft = fields.fft;
% rec_info.fft.FOV = fields.rec.Size*fields.fbp.MaxGradient*2.802;
rec_info.rec = fields.spi;
rec_info.rec = copy_fields(rec_info.rec, fields.rec);
rec_info.rec.Enabled = epr_if(strcmp(safeget(fields.prc, 'recon_data','yes'),'yes'), 'on', 'off');
rec_info.prc = fields.prc;

% Pulse data do not have Hardware calibration
fields.clb.ampHH = 1;

% identify data source
if isstruct(file_name)
  FinalImage.raw_info = file_name.raw_info;
  mat      = file_name.mat;
  mat_bl   = file_name.mat_bl;
  is_process_data = is_recon_data;
  [fp, fn]=fileparts(FinalImage.raw_info.FileName);
else
  % select file names
  if iscell(file_name), [fp, fn, fext]=fileparts(file_name{1});
  else [fp, fn, fext]=fileparts(file_name);
  end
  
  switch fext
    case {'.d01', '.tdms'}
      load_opt.Modality = 'PULSESPI';
      load_opt.Sequence  = 'FIDSRT';
      if iscell(file_name)
        for ii=1:length(file_name)
          [mat_1,mat_bl_1,FinalImage.raw_info]=epr_ReadPulseImageFile(file_name{ii}, load_opt);
          if ii==1, mat = mat_1; mat_bl = mat_bl_1;
          else mat = mat + mat_1; mat_bl = mat_bl + mat_bl_1;
          end
        end
        mat = mat / length(file_name); mat_bl = mat_bl / length(file_name);
      else
        [mat,mat_bl,FinalImage.raw_info]=epr_ReadPulseImageFile(file_name, load_opt);
      end
      is_process_data = 1;
    case '.mat'
      s = load(file_name);
      FinalImage.raw_info = s.raw_info;
      mat      = s.mat;
      mat_bl   = s.mat_bl;
      is_process_data = is_recon_data;
  end
end
disp('    Loaded.');

% Prepare file names
if isempty(output_path)
  savefile = fullfile(fp, [fn, file_suffix,'.mat']);
  psavefile = fullfile(fp, ['p',fn, file_suffix,'.mat']);
else
  savefile = fullfile(output_path, [fn, file_suffix, '.mat']);
  psavefile = fullfile(output_path, ['p', fn, file_suffix, '.mat']);
end

if is_process_data
  if isfield(rec_info.rec, 'recon_echos') && ~isempty(rec_info.rec.recon_echos)
    list_of_echoes = rec_info.rec.recon_echos;
    if isfield(rec_info.td, 'use_echos') && ~isempty(rec_info.td.use_echos)
      idx = zeros(length(FinalImage.raw_info.tau2), 1);
      idx(rec_info.td.use_echos) = 1;
      rec_info.td.use_echos = find(idx(list_of_echoes));
    end
    FinalImage.raw_info.T = FinalImage.raw_info.T(list_of_echoes);
  else
    list_of_echoes = 1:size(mat, 2);
  end
  yyy = mat - mat_bl;
  
  FinalImage.Raw = [];
  % run reconstruction for every echo --------------------------------------
  image_str = cell(numel(list_of_echoes), 1);
  for echo_n = list_of_echoes
    image_str{echo_n} = ['Image ', num2str(echo_n)];
  end
  
  FinalImage.rec_info = rec_info;
  
  tm = FinalImage.raw_info.t_ax*1E6 + rec_info.td.acq_window_start*1E-3; % us
  SPIdelay = rec_info.rec.SPIdelay * 1E-3*ones(length(list_of_echoes), 1);
  
  gidx = FinalImage.raw_info.gidx(FinalImage.raw_info.gidx ~= 0);
  Gmax = FinalImage.raw_info.MaxGradient;
  
  for pnt_n=1:length(list_of_echoes)
    [mm, idx] = min((tm-SPIdelay(pnt_n)).^2);
    tp = tm(idx);
    FinalImage.rec_info.Size = FinalImage.raw_info.Dim/2/tp/Gmax/2.8.*(1+2./FinalImage.raw_info.Dim);
    disp(sprintf('Selected point: %i (%f us)', idx, SPIdelay(pnt_n)))
    yTP = zeros(FinalImage.raw_info.Dim);
    
    for ii=1:size(mat, 3)
      yyy(:, pnt_n, ii) = smooth(yyy(:, pnt_n, ii), 7);
    end
    yTP(gidx) = squeeze(yyy(idx, pnt_n ,:));

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
    
    yTP = yTP * exp(1i*rec_info.td.phase_zero_order*pi/180);
    %    right_phase = rec_info.td.phase_zero_order*pi/180*ones(length(use_echos),sz(3));
    
    %   yTP = squeeze(sum(yy(idx-1:idx+1,:,:,:),1));
    %
    %   disp(handles.mat_info.Dim/2/tp/Gmax/2.8)
    switch 'as_is'
      case 'as_is'
        yTPzero = zeros(FinalImage.raw_info.Dim*3);
        yTPzero(FinalImage.raw_info.Dim(1)+1:2*FinalImage.raw_info.Dim(1),...
          FinalImage.raw_info.Dim(2)+1:2*FinalImage.raw_info.Dim(2),...
          FinalImage.raw_info.Dim(3)+1:2*FinalImage.raw_info.Dim(3)) = yTP;
        FinalImage.Raw(:,:,:,pnt_n) = real(fftshift(ifftn(ifftshift(yTPzero))));
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
    %   if usepause, pause; else drawnow; end
  end
  
  if strcmpi(safeget(fields.fft, 'fft_export_clearence', 'no'), 'yes')
    FinalImage.Clearence = safeget(dsc, 'export_clearence', []);
  end
elseif exist('s', 'var')
  FinalImage.Raw = s.mat_recFXD;
  FinalImage.rec_info = s.rec_info;
  is_recon_data = 1;
end

if ndims(FinalImage.Raw) ~=4, is_fit_data = false; end
if is_fit_data
  [FinalImage.fit_data] = td_recovery_fit(FinalImage.Raw, FinalImage.raw_info.Trep(:)'*1E6, fields.fit);
  
  [FinalImage.Amp, FinalImage.T2, FinalImage.Mask, FinalImage.Error, FinalImage.Error_T2] = LoadFitPars(FinalImage.fit_data, {'Amp','T2','Mask','Error', 'Error_T2'});
  FinalImage.pO2 = epr_T2_PO2(FinalImage.T2, FinalImage.Amp, FinalImage.Mask, fields.clb);
  %       Q_correction = sqrt(s1.mat_pO2_info.Qcb/s1.mat_pO2_info.Q);
  Q_correction = 1;
  FinalImage.Amp = FinalImage.Amp * Q_correction/ fields.clb.amp1mM;
else
  FinalImage.fit_data = [];
end

if strcmp(safeget(rec_info.prc, 'save_data','yes'),'yes')
  s.file_type    = 'Image_v1.1';
  s.raw_info     = FinalImage.raw_info;
  s.mat          = single(mat);
  s.mat_bl       = single(mat_bl);
  if is_recon_data
    s.mat_recFXD   = single(FinalImage.Raw);
    s.rec_info = FinalImage.rec_info;
    s.pO2_info = fields.clb;
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
  
  td_CreateDirectory(savefile);
  if is_process_data
    save(savefile,'-struct','s');
  end
  if is_fit_data
    save(psavefile,'-struct','s1');
  end
  disp(sprintf('    Data are saved to %s.', psavefile));
end

% --------------------------------------------------------------------
function s = copy_fields(s, s1)

fname = fieldnames(s1);
for ii=1:length(fname), s.(fname{ii}) = s1.(fname{ii}); end

