% function Image = epr_LoadMATFile(fname, varargin)
% Load image from MAT file. Supports CW and ESE image formats
% varargin{1} can be false/true or the calibration structure.
%    when false calibration parameters will be taken from file
%    when true calibration parameters dialog will appear
%    when calibration structure supplied fields other than supplied will
%    have  default value
%    Torr_per_mGauss, LLW_zero_po2, mG_per_mM, amp1mM, ampHH, Q, Qcb
% varargin{2} cell array of output parameters (case insensitive)
%    these can be              {'CC', 'pO2', 'ERROR', 'RAWINT' }
%    for CW  these also can be {'SNR', 'LW', 'PHASE', 'XOVER'}
%    for ESE these also can be {'T2', 'LW', 'RAW', 'RAW3D'}
%    if ommited:               {'CC', 'pO2'}
%   the output will be a struct with corresponding fields
%   OR
% varargin{2} single string with one output parameter
%   the output will be the image itself
% units:
%    CC  [mM]
%    pO2 [torr]
%    LW  [mG]
%    T2  [us]
% Example2:
%    Image = epr_LoadMATFile('C:\my_image.mat', true, {'CC', 'pO2', 'LW'});
%    pO2 = epr_LoadMATFile('C:\my_image.mat', true, 'pO2');

function Image = epr_LoadMATFile(fname, varargin)

if ~exist('fname', 'var')
  help epr_LoadMATFile;
  return;
elseif nargin >= 2,
  if islogical(varargin{1})
    ConstDialog = varargin{1};
  else
    mat_pO2_info = varargin{1};
    ConstDialog = safeget(mat_pO2_info, 'Dialog', false);
  end
else
  ConstDialog = false;
end

isSingleOutput = 0;
AdditionalOutputsCW = {'PO2', 'CC'};
AdditionalOutputsESE = {'PO2', 'CC'};
if nargin >=3
  if isstruct(varargin{2})
    AdditionalOutputsCW = safeget(varargin{2}, 'CW', AdditionalOutputsCW);
    AdditionalOutputsESE = safeget(varargin{2}, 'ESE', AdditionalOutputsESE);
  else
    if iscell(varargin{2})
      AdditionalOutputsCW = varargin{2};
      AdditionalOutputsESE = varargin{2};
    else
      isSingleOutput = 1;
      AdditionalOutputsCW = varargin(2);
      AdditionalOutputsESE = varargin(2);
    end
  end
end

LOAD_PO2 = 1; LOAD_CC = 2; LOAD_RAW = 3;
LOAD_ERROR = 4; LOAD_SNR = 5; LOAD_LW = 6; LOAD_T2 = 7; LOAD_T1 = 8;
LOAD_RAWFIT = 9; LOAD_RAW3D = 10; LOAD_PHASE = 11;
LOAD_XOVR = 12; LOAD_PRJ = 13; LOAD_ERR_T2 = 14; LOAD_ERR_T1 = 15;
LOAD_R1 = 16;
LOAD_LAST = 17;

Image.FileName = fname; Image.Size = [];
if isempty(fname), Image.Mask = []; return; end

s1 = load(strtrim(fname));
disp(['Image ', Image.FileName, ' is loaded.']);

if isfield(s1, 'mat_rec_info'), s1.rec_info = s1.mat_rec_info; end

file_type = safeget(s1, 'file_type', '+++');
if strcmp(file_type, '+++')
  if isfield(s1, 'mat_bl'), file_type = 'Image_v0.1';
  elseif isfield(s1, 'P'), file_type = 'ImageCW_v1.0';
  else file_type = 'ImageCW_v0.9';
  end
end

switch file_type
  case 'ArbuzGeneric_v1.0'
    Image.Raw  = s1.data;
    Image.Size = size(Image.Raw);
  case 'ImageCW_v0.9'
    add_out = load_options(AdditionalOutputsCW);
    if add_out(LOAD_RAW)
      Image.Raw  = s1.mat_recFXD;
    else
      Image.Raw  = epr_LoadSpatialImage(s1.mat_recFXD);
    end
    Image.Size = s1.pars_out(8)*[1,1,1];
    Image.raw_info.pars_out = s1.pars_out;
  case 'ImageCW_v1.0'
    add_out = load_options(AdditionalOutputsCW);
    Image.raw_info.data.FBP = struct('nAz', s1.pars_out(5), 'nPolar', s1.pars_out(6),...
      'nSpec', 14, 'imtype', 1, 'MaxGradient', 3.028,...
      'angle_sampling', 'UNIFORM_SPATIAL_FLIP');
    pars = td_GetFBPGradientTable(Image.raw_info.data.FBP);
    Image.raw_info = epr_CopyFields(Image.raw_info, pars, {'gidx','GradX','GradY','GradZ','nP','nTrace','Dim'});
    Image.raw_info.FieldSweep = s1.pars_out(19) * pars.UnitSweep;
    Image.raw_info.data.Modality = 'CWFBP';
    
    if isfield(s1, 'immask')
      if add_out(LOAD_CC) || add_out(LOAD_PO2) || add_out(LOAD_ERROR) || ...
          add_out(LOAD_LW) || add_out(LOAD_RAWFIT) || add_out(LOAD_SNR)
        % Load fitted data
        idx = permute(s1.immask > 0, [3,2,1]);
        if add_out(LOAD_PO2) || add_out(LOAD_CC) || add_out(LOAD_RAWFIT), raw_int = load_p(idx, s1.xtra_info(2,:)); end
        if add_out(LOAD_PO2) || add_out(LOAD_LW), lw = load_p(idx, s1.P(2,:)); end
        if add_out(LOAD_LW), Image.LW = lw*1E3; end
        if add_out(LOAD_RAWFIT), Image.RAW_INT = raw_int; end
        if add_out(LOAD_ERROR), Image.Error = load_p(idx, s1.P_errs(2,:)); end
        if add_out(LOAD_SNR), Image.SNR = load_p(idx, s1.xtra_info(7,:)); end
        if add_out(LOAD_PHASE), Image.PHASE = load_p(idx, s1.P(7,:)); end
        if add_out(LOAD_XOVR), Image.XOVER = load_p(idx, s1.P(1,:)); end
        
        % Copy all parameters from the given structure
        if exist('mat_pO2_info', 'var')
          s1.mat_pO2_info = check_mat_pO2_info([], mat_pO2_info);
        else
          s1.mat_pO2_info = check_mat_pO2_info([], []);
        end
        
        % manual selection of parameters
        if ConstDialog
          [calb, calb_idx] = lab_calibration('CWEPROI', []);
          n = listdlg('PromptString','Select a calibration:','SelectionMode','single', ...
            'ListSize', [220, 300], 'ListString', calb);
          if isempty(n), n=1; end
          s1.mat_pO2_info = lab_calibration(calb_idx(n));
          s1.mat_pO2_info = check_mat_pO2_info([],s1.mat_pO2_info);
        elseif ~exist('mat_pO2_info', 'var')
          s1.mat_pO2_info.Q = safeget(s1.mat_pO2_info, 'Q', 300);
          s1.mat_pO2_info.Qcb = safeget(s1.mat_pO2_info, 'Qcb', s1.mat_pO2_info.Q);
        end
        s1.mat_pO2_info.ModBroadn_0 = safeget(s1.mat_pO2_info, 'ModBroadn_0', 0.1714);
        s1.mat_pO2_info.ModBroadn_1 = safeget(s1.mat_pO2_info, 'ModBroadn_1', 1);
        
        if ConstDialog
          s1.mat_pO2_info = EditConstDialog(s1.mat_pO2_info);
        else
          ret = inputdlg({'Image quality factor'; 'Calibration quality factor'}, 'Input', 1, {num2str(s1.mat_pO2_info.Qcb),num2str(s1.mat_pO2_info.Qcb)});
          s1.mat_pO2_info.Q = str2double(ret{1});
          s1.mat_pO2_info.Qcb = str2double(ret{2});
        end
        s1.mat_pO2_info.ModCalImage = str2double(s1.cpv(9,:));
        
        Image.Mask = s1.immask > 0;
        if exist('raw_int','var'), Image.Mask = Image.Mask & ~isnan(raw_int); end
        Image.Amp = Int2ConcentrationCW(raw_int, s1.pars_out, s1.mat_pO2_info);
        Image.pO2 = (lw*1e3 - s1.mat_pO2_info.LLW_zero_po2 - Image.Amp*s1.mat_pO2_info.mG_per_mM) * s1.mat_pO2_info.Torr_per_mGauss;
        
        if isfield(Image, 'Error')
          Image.max_Error = max(Image.Error(Image.Mask(:)));
          Image.min_Error = min(Image.Error(Image.Mask(:)));
        end
        
        Image.Size = s1.pars_out(8)*[1,1,1];
        Image.SourceFileName = sscanf(s1.cpv(1,:),'%s');
        Image.pO2_info = s1.mat_pO2_info;
        if isfield(Image, 'raw_info'), Image.raw_info = Image.raw_info; end
        Image.raw_info.data.Modality = 'CWFBP';
      end
    else
      % switch on RAW format
      add_out(LOAD_RAW) = 1;
    end
    
    % raw data
    if add_out(LOAD_RAW) || add_out(LOAD_RAW3D) || add_out(LOAD_PRJ)
      [fpath, fname, fext] = fileparts(fname);
      raw_file_name = fullfile(fpath, [fname(2:end), fext]);
      if ~exist(raw_file_name, 'file')
        [raw_file_name, raw_path] = uigetfile({'*.mat', 'MATLAB File (*.mat)'; '*.*', 'All Files (*.*)'}, ...
          'Pick a 4D file', raw_file_name);
        if ~(isequal(raw_file_name,0) || isequal(raw_path,0))
          raw_file_name = fullfile(raw_path, raw_file_name);
        end
      end
      s2 = load(raw_file_name);
      if add_out(LOAD_RAW)
        Image.Raw  = s2.mat_recFXD;
      elseif add_out(LOAD_RAW3D)
        Image.Raw  = epr_LoadSpatialImage(s2.mat_recFXD);
      elseif add_out(LOAD_PRJ)
        gidx = false(Image.raw_info.Dim);
        gidx(Image.raw_info.gidx) = true;
        Image.Prj = s2.mat(:, gidx);
        service_fld = fix(Image.Prj/1E6);
        scans = abs(service_fld); scans(scans==0)=1;
        Image.Prj = (Image.Prj - 1E6 * service_fld)./scans;
        Image.Prj(service_fld == 0) = 0;
        Image.Prj = cumsum(Image.Prj(), 1);
      end
      Image.Size = s2.pars_out(8)*[1,1,1];
      Image.raw_info.pars_out = s2.pars_out;
      Image.raw_info.data.Hardware.mod_cal = str2double(s1.cpv(9,:));
      Image.raw_info.ModFrequency = Image.raw_info.pars_out(17)*1E-3/2.802;
      %       if isfield(s1.mat_pO2_info, 'ModCalImage')
      %         Image.raw_info.ModAmplitude = Image.raw_info.pars_out(16)*Image.pO2_info.ModCalImage;
      %       else
      Image.raw_info.ModAmplitude = Image.raw_info.pars_out(16)*Image.raw_info.data.Hardware.mod_cal;
      %       end
      Image.rec_info.rec.deltaH = Image.raw_info.pars_out(19);
    end
  case 'Image_v1.0'
    add_out = load_options(AdditionalOutputsESE);
    mat_info = safeget(s1, 'mat_info', []);
    Data = safeget(mat_info, 'data', []);
    Modality = safeget(Data, 'Modality', 'PULSEFBP');
    switch upper(Modality)
      case 'PULSEFBP'
        if add_out(LOAD_CC) || add_out(LOAD_PO2) || add_out(LOAD_ERROR) || ...
            add_out(LOAD_RAWFIT) || add_out(LOAD_T2)
          if add_out(LOAD_ERROR)
            [Image.Amp, T2, Image.Mask, Image.Error] = LoadFitPars(safeget(s1, 'mat_fit', ''),...
              {'Amp','T2','Mask','Error'});
          else
            [Image.Amp, T2, Image.Mask] = LoadFitPars(safeget(s1, 'mat_fit', ''));
            Image.Error = [];
          end
          if add_out(LOAD_T2), Image.T2 = T2; end
          if add_out(LOAD_LW), Image.LW = epr_T2_LW(T2, Image.Mask); end
          
          Image.StartTime = safeget(mat_info, 'StartTime', 0);
          Image.FinishTime = safeget(mat_info, 'FinishTime', 0);
          Image.SourceFileName = safeget(s1, 'name_com', '');
          Image.Size = s1.mat_rec_info.rec.Size;
          Image.rec_info = s1.mat_rec_info;
          
          if ~isempty(T2) && (add_out(LOAD_CC) || add_out(LOAD_PO2) || ...
              add_out(LOAD_RAWFIT))
            
            %             s1.mat_pO2_info.llw_zero_po2 = 12.4;
            % Homogeneous phantom code
            %             fake_amp_map = s1.mat_pO2_info.amp1mM *ones(size(Image.Amp));
            %             Image.pO2 = td_T2_PO2(T2, fake_amp_map, Image.Mask, s1.mat_pO2_info);
            
            % Copy all parameters from the given structure
            if exist('mat_pO2_info', 'var')
              s1.mat_pO2_info = check_mat_pO2_info(s1.mat_pO2_info, mat_pO2_info);
            else
              s1.mat_pO2_info = check_mat_pO2_info([], s1.mat_pO2_info);
            end
            
            % manual selection of parameters
            if ConstDialog
              s1.mat_pO2_info = EditConstDialog(s1.mat_pO2_info);
            elseif ~exist('mat_pO2_info', 'var')
              s1.mat_pO2_info.Q = safeget(s1.mat_pO2_info, 'Q', 15);
              s1.mat_pO2_info.Qcb = safeget(s1.mat_pO2_info, 'Qcb', s1.mat_pO2_info.Q);
            end
            
            Q_correction = sqrt(s1.mat_pO2_info.Qcb/s1.mat_pO2_info.Q);
            
            %  Image
            Image.pO2 = epr_T2_PO2(T2, Image.Amp, Image.Mask, s1.mat_pO2_info);
            Image.Amp = Image.Amp * Q_correction/ s1.mat_pO2_info.amp1mM;
            Image.pO2_info = s1.mat_pO2_info;
            Image.raw_info = s1.mat_info;
          end
        end
        if add_out(LOAD_RAW)
          Image.Raw  = s1.mat_recFXD;
          Image.Size = s1.mat_rec_info.rec.Size;
          Image.raw_info = s1.mat_info;
        elseif add_out(LOAD_RAW3D)
          Image.Raw  = epr_LoadSpatialImage(s1.mat_recFXD);
          Image.Size = s1.mat_rec_info.rec.Size;
        end
      case 'RS_FBP'
        if add_out(LOAD_CC) || add_out(LOAD_PO2) || add_out(LOAD_ERROR) || ...
            add_out(LOAD_RAWFIT) || add_out(LOAD_T2)
          if add_out(LOAD_ERROR)
            [Image.Amp, T2, Image.Mask, Image.Error] = LoadFitPars(safeget(s1, 'mat_fit', ''),...
              {'Amp','T2','Mask','Error'});
          else
            [Image.Amp, T2, Image.Mask] = LoadFitPars(safeget(s1, 'mat_fit', ''));
            Image.Error = [];
          end
          if add_out(LOAD_T2), Image.T2 = T2; end
          if add_out(LOAD_LW), Image.LW = epr_T2_LW(T2, Image.Mask); end
          
          Image.StartTime = safeget(mat_info, 'StartTime', 0);
          Image.FinishTime = safeget(mat_info, 'FinishTime', 0);
          Image.SourceFileName = '';
          Image.Size = s1.mat_rec_info.rec.Size;
          
          if ~isempty(T2) && (add_out(LOAD_CC) || add_out(LOAD_PO2) || ...
              add_out(LOAD_RAWFIT))
            
            %             s1.mat_pO2_info.llw_zero_po2 = 12.4;
            % Homogeneous phantom code
            %             fake_amp_map = s1.mat_pO2_info.amp1mM *ones(size(Image.Amp));
            %             Image.pO2 = td_T2_PO2(T2, fake_amp_map, Image.Mask, s1.mat_pO2_info);
            
            % Copy all parameters from the given structure
            if exist('mat_pO2_info', 'var')
              s1.mat_pO2_info = check_mat_pO2_info(s1.mat_pO2_info, mat_pO2_info);
            else
              s1.mat_pO2_info = check_mat_pO2_info([], s1.mat_pO2_info);
            end
            
            % manual selection of parameters
            if ConstDialog
              s1.mat_pO2_info = EditConstDialog(s1.mat_pO2_info);
            elseif ~exist('mat_pO2_info', 'var')
              s1.mat_pO2_info.Q = safeget(s1.mat_pO2_info, 'Q', 15);
              s1.mat_pO2_info.Qcb = safeget(s1.mat_pO2_info, 'Qcb', s1.mat_pO2_info.Q);
            end
            
            Q_correction = sqrt(s1.mat_pO2_info.Qcb/s1.mat_pO2_info.Q);
            
            %  Image
            Image.pO2 = epr_T2_PO2(T2, Image.Amp, Image.Mask, s1.mat_pO2_info);
            Image.Amp = Image.Amp * Q_correction/ s1.mat_pO2_info.amp1mM;
            Image.pO2_info = s1.mat_pO2_info;
            Image.raw_info = s1.mat_info;
          end
        end
        if add_out(LOAD_RAW)
          Image.Raw  = s1.mat_recFXD;
          Image.rec_info = s1.mat_rec_info;
          Image.Size = s1.mat_rec_info.rec.Size;
          Image.raw_info = s1.mat_info;
        elseif add_out(LOAD_RAW3D)
          Image.Raw  = epr_LoadSpatialImage(s1.mat_recFXD);
          Image.Size = s1.mat_rec_info.rec.Size;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%                                   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%        %%%                %%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%      %%%%%              %%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%     %%%%%%             %%%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%    %%%%%%%           %%%%%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%   %%%%%%%%          %%%%%%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%       %%%%              %%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%       %%%%              %%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%       %%%%              %%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%       %%%%              %%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%       %%%%      %%      %%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%    %%%%%%%%%%   %%   %%%%%%%%%%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%                                   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  case 'Image_v1.1'
    add_out = load_options(AdditionalOutputsCW);
    raw_info = safeget(s1, 'raw_info', []);
    Data = safeget(raw_info, 'data', []);
    Modality = upper(safeget(Data, 'Modality', 'PULSEFBP'));
    switch Modality
      case 'PULSEFBP'
        if add_out(LOAD_CC) || add_out(LOAD_PO2) || add_out(LOAD_ERROR) || ...
            add_out(LOAD_RAWFIT) || add_out(LOAD_T2)
          if add_out(LOAD_ERROR)
            [Image.Amp, T2, Image.Mask, Image.Error] = LoadFitPars(safeget(s1, 'mat_fit', ''),...
              {'Amp','T2','Mask','Error'});
          else
            [Image.Amp, T2, Image.Mask] = LoadFitPars(safeget(s1, 'mat_fit', ''));
            Image.Error = [];
          end
          if add_out(LOAD_T2), Image.T2 = T2; end
          if add_out(LOAD_LW), Image.LW = epr_T2_LW(T2, Image.Mask); end
          
          Image.StartTime = safeget(raw_info, 'StartTime', 0);
          Image.FinishTime = safeget(raw_info, 'FinishTime', 0);
          Image.SourceFileName = raw_info.FileName;
          Image.Size = s1.rec_info.rec.Size;
          
          if ~isempty(T2) && (add_out(LOAD_CC) || add_out(LOAD_PO2) || ...
              add_out(LOAD_RAWFIT))
            
            %             s1.mat_pO2_info.llw_zero_po2 = 12.4;
            % Homogeneous phantom code
            %             fake_amp_map = s1.mat_pO2_info.amp1mM *ones(size(Image.Amp));
            %             Image.pO2 = td_T2_PO2(T2, fake_amp_map, Image.Mask, s1.mat_pO2_info);
            
            % Copy all parameters from the given structure
            if exist('mat_pO2_info', 'var')
              s1.mat_pO2_info = check_mat_pO2_info(s1.mat_pO2_info, mat_pO2_info);
            else
              s1.mat_pO2_info = check_mat_pO2_info([], s1.mat_pO2_info);
            end
            
            % manual selection of parameters
            if ConstDialog
              s1.mat_pO2_info = EditConstDialog(s1.mat_pO2_info);
            elseif ~exist('mat_pO2_info', 'var')
              s1.mat_pO2_info.Q = safeget(s1.mat_pO2_info, 'Q', 15);
              s1.mat_pO2_info.Qcb = safeget(s1.mat_pO2_info, 'Qcb', s1.mat_pO2_info.Q);
            end
            
            Q_correction = sqrt(s1.mat_pO2_info.Qcb/s1.mat_pO2_info.Q);
            
            %  Image
            Image.pO2 = epr_T2_PO2(T2, Image.Amp, Image.Mask, s1.mat_pO2_info);
            Image.Amp = Image.Amp * Q_correction/ s1.mat_pO2_info.amp1mM;
            Image.pO2_info = s1.mat_pO2_info;
            Image.raw_info = s1.mat_info;
          end
        end
        if add_out(LOAD_RAW)
          Image.Raw  = s1.mat_recFXD;
          Image.Size = s1.rec_info.rec.Size;
          Image.raw_info = s1.rec_info;
        elseif add_out(LOAD_RAW3D)
          Image.Raw  = epr_LoadSpatialImage(s1.mat_recFXD);
          Image.Size = s1.rec_info.rec.Size;
        end
      case 'RSFBP'
        if add_out(LOAD_CC) || add_out(LOAD_PO2) || add_out(LOAD_ERROR) || ...
            add_out(LOAD_RAWFIT) || add_out(LOAD_T2)
          if add_out(LOAD_ERROR)
            [Image.Amp, T2, Image.Mask, Image.Error] = LoadFitPars(safeget(s1, 'mat_fit', ''),...
              {'Amp','T2','Mask','Error'});
          else
            [Image.Amp, T2, Image.Mask] = LoadFitPars(safeget(s1, 'mat_fit', ''));
            Image.Error = [];
          end
          if add_out(LOAD_T2), Image.T2 = T2; end
          if add_out(LOAD_LW), Image.LW = epr_T2_LW(T2, Image.Mask); end
          
          Image.StartTime = safeget(raw_info, 'StartTime', 0);
          Image.FinishTime = safeget(raw_info, 'FinishTime', 0);
          Image.SourceFileName = '';
          Image.Size = s1.rec_info.rec.Size;
          Image.rec_info = s1.rec_info;
          
          if ~isempty(T2) && (add_out(LOAD_CC) || add_out(LOAD_PO2) || ...
              add_out(LOAD_RAWFIT))
            
            %             s1.mat_pO2_info.llw_zero_po2 = 12.4;
            % Homogeneous phantom code
            %             fake_amp_map = s1.mat_pO2_info.amp1mM *ones(size(Image.Amp));
            %             Image.pO2 = td_T2_PO2(T2, fake_amp_map, Image.Mask, s1.mat_pO2_info);
            
            % Copy all parameters from the given structure
            if exist('mat_pO2_info', 'var')
              s1.mat_pO2_info = check_mat_pO2_info(s1.mat_pO2_info, mat_pO2_info);
            else
              s1.mat_pO2_info = check_mat_pO2_info([], s1.mat_pO2_info);
            end
            
            % manual selection of parameters
            if ConstDialog
              s1.mat_pO2_info = EditConstDialog(s1.mat_pO2_info);
            elseif ~exist('mat_pO2_info', 'var')
              s1.mat_pO2_info.Q = safeget(s1.mat_pO2_info, 'Q', 15);
              s1.mat_pO2_info.Qcb = safeget(s1.mat_pO2_info, 'Qcb', s1.mat_pO2_info.Q);
            end
            
            Q_correction = sqrt(s1.mat_pO2_info.Qcb/s1.mat_pO2_info.Q);
            
            %  Image
            Image.pO2 = epr_T2_PO2(T2, Image.Amp, Image.Mask, s1.mat_pO2_info);
            Image.Amp = Image.Amp * Q_correction/ s1.mat_pO2_info.amp1mM;
          end
        end
        if isempty(Image.Size)
        end
        if add_out(LOAD_RAW)
          Image.Raw  = s1.mat_recFXD;
          Image.rec_info = s1.rec_info;
          Image.Size = s1.rec_info.rec.Size;
          Image.raw_info = s1.raw_info;
        elseif add_out(LOAD_RAW3D)
          Image.Raw  = epr_LoadSpatialImage(s1.mat_recFXD);
          Image.Size = s1.rec_info.rec.Size;
        end
        if add_out(LOAD_PRJ)
          Image.Prj = s1.mat;
        end
        Image.pO2_info = s1.pO2_info;
        Image.raw_info = s1.raw_info;
      case 'CWFBP'
        add_out = load_options(AdditionalOutputsCW);
        Image.Size = s1.rec_info.rec.Size*[1,1,1];
        Image.raw_info = s1.raw_info;
        if add_out(LOAD_RAW)
          Image.Raw  = s1.mat_recFXD;
          Image.rec_info = s1.rec_info;
          Image.Size = s1.rec_info.rec.Size;
          if isfield(Image, 'raw_info')
            Image.raw_info = s1.raw_info;
          else
            Image.raw_info = s1.mat_info;
          end
        elseif add_out(LOAD_RAW3D)
          Image.Raw  = epr_LoadSpatialImage(s1.mat_recFXD);
          Image.Size = s1.rec_info.rec.Size;
        end
        if add_out(LOAD_PRJ)
            pars = td_GetFBPGradientTable(s1.raw_info.data.FBP);
            Image.raw_info = epr_CopyFields(Image.raw_info, pars, {'gidx','GradX','GradY','GradZ','nP','nTrace','Dim'});
            Image.raw_info.FieldSweep = 1.024 *sqrt(2)* pars.UnitSweep;
            gidx = false(pars.Dim);
            gidx(pars.gidx) = true;
            Image.Prj = s1.mat(:, gidx);
            service_fld = fix(Image.Prj/1E6);
            scans = abs(service_fld); scans(scans==0)=1;
            Image.Prj = (Image.Prj - 1E6 * service_fld)./scans;
            Image.Prj(service_fld == 0) = 0;
            Image.Prj = cumsum(Image.Prj(), 1);
        end
    end
  case 'Image_v0.1'
    add_out = load_options(AdditionalOutputsESE);
    s1.mat_pO2_info = check_mat_pO2_info([], s1.mat_pO2_info);
    Image.Amp = s1.mat_recAmp;
    Image.Mask = s1.mat_recT2 > 0.1;
    Image.pO2 = epr_T2_PO2(s1.mat_recT2, Image.Amp, Image.Mask, s1.mat_pO2_info);
    if add_out(LOAD_T2), Image.T2 = s1.mat_recT2; end
    Image.Size = s1.mat_rec_info.rec.Size*[1,1,1];
    if add_out(LOAD_RAW)
      Image.Raw  = s1.mat_recFXD;
      Image.Size = s1.mat_rec_info.rec.Size;
      Image.raw_info = s1.mat_info;
    elseif add_out(LOAD_RAW3D)
      Image.Raw  = epr_LoadSpatialImage(s1.mat_recFXD);
      Image.Size = s1.mat_rec_info.rec.Size;
    end
  case 'FitImage_v1.0'
    Image.rec_info = s1.mat_rec_info;
    switch s1.mat_fit.Algorithm
      case 'CW_spectral_fit_R2_XOVR_PHASE'
        add_out = load_options(AdditionalOutputsCW);
        
        % Copy all parameters from the given structure
        if exist('mat_pO2_info', 'var')
          s1.mat_pO2_info = check_mat_pO2_info(s1.mat_pO2_info, mat_pO2_info);
        else
          s1.mat_pO2_info = check_mat_pO2_info([], s1.mat_pO2_info);
        end
        
        % manual selection of parameters
        if ConstDialog
          s1.mat_pO2_info = EditConstDialog(s1.mat_pO2_info);
        elseif ~exist('mat_pO2_info', 'var')
          s1.mat_pO2_info.Q = safeget(s1.mat_pO2_info, 'Q', 15);
          s1.mat_pO2_info.Qcb = safeget(s1.mat_pO2_info, 'Qcb', s1.mat_pO2_info.Q);
        end
        
        if add_out(LOAD_CC) || add_out(LOAD_RAWFIT) || add_out(LOAD_PO2), raw_int = LoadFitPars(s1.mat_fit, {'Amp'} ); end
        if add_out(LOAD_PO2) || add_out(LOAD_LW), lw = LoadFitPars(s1.mat_fit, {'LLW'} ); end
        if add_out(LOAD_LW), Image.LW = lw*1E3; end
        if add_out(LOAD_RAWFIT), Image.RAW_INT = raw_int; end
        %         if add_out(LOAD_ERROR), Image.Error = load_p(idx, s1.P_errs(2,:)); end
        %         if add_out(LOAD_PHASE), Image.PHASE = load_p(idx, s1.P(7,:)); end
        %         if add_out(LOAD_XOVR), Image.XOVER = load_p(idx, s1.P(1,:)); end
        Image.Size = s1.mat_rec_info.rec.Size;
        Image.pO2_info = s1.mat_pO2_info;
        Image.Mask = LoadFitPars(s1.mat_fit, {'Mask'} );
        if add_out(LOAD_CC) || add_out(LOAD_PO2)
          raw_int = raw_int*safeget(Image.pO2_info, 'ampHH', 1);
          
          % quality factor correction
          Q_correction = sqrt(Image.pO2_info.Qcb/Image.pO2_info.Q);
          Image.Amp = raw_int * Q_correction/ Image.pO2_info.amp1mM;
          Image.Mask = Image.Mask & Image.Amp > 0;
        end
        if add_out(LOAD_PO2)
          Image.pO2 = epr_LLW_PO2(lw*1E3, raw_int, Image.Mask, Image.pO2_info);
        end
        if add_out(LOAD_XOVR), Image.XOVER = LoadFitPars(s1.mat_fit, {'xover'} ); end
        if add_out(LOAD_PHASE), Image.PHASE = LoadFitPars(s1.mat_fit, {'phase'} ); end
        if add_out(LOAD_ERROR), Image.Error = LoadFitPars(s1.mat_fit, {'error'} ); end
        % raw data
        Image.raw_info = s1.raw_info;
        Image.rec_info = s1.mat_rec_info;
        if add_out(LOAD_RAW) || add_out(LOAD_RAW3D)
          [fpath, fname1, fext] = fileparts(fname);
          raw_file_name = fullfile(fpath, [fname1(2:end), fext]);
          if ~exist(raw_file_name, 'file')
            [raw_file_name, raw_path] = uigetfile({'*.mat', 'MATLAB File (*.mat)'; '*.*', 'All Files (*.*)'}, ...
              'Pick a 4D file', raw_file_name);
            if ~(isequal(raw_file_name,0) || isequal(raw_path,0))
              raw_file_name = fullfile(raw_path, raw_file_name);
            end
          end
          s2 = load(raw_file_name);
          if add_out(LOAD_RAW)
            Image.Raw  = s2.mat_recFXD;
          else
            Image.Raw  = epr_LoadSpatialImage(s2.mat_recFXD);
          end
          Image.raw_info = s2.raw_info;
        end
      case 'T2_ExpDecay_No_Offset'
        add_out = load_options(AdditionalOutputsESE);
        
        % Copy all parameters from the given structure
        if exist('mat_pO2_info', 'var')
          s1.mat_pO2_info = check_mat_pO2_info(s1.mat_pO2_info, mat_pO2_info);
        else
          s1.mat_pO2_info = check_mat_pO2_info([], s1.mat_pO2_info);
        end
        
        if add_out(LOAD_CC) || add_out(LOAD_PO2) || add_out(LOAD_ERROR) || ...
            add_out(LOAD_RAWFIT) || add_out(LOAD_T2)
          if add_out(LOAD_ERROR)
            [Image.Amp, T2, Image.Mask, Image.Error] = LoadFitPars(safeget(s1, 'mat_fit', ''),...
              {'Amp','T2','Mask','Error'});
          else
            [Image.Amp, T2, Image.Mask] = LoadFitPars(safeget(s1, 'mat_fit', ''));
            Image.Error = [];
          end
          if add_out(LOAD_T2), Image.T2 = T2; end
          if add_out(LOAD_LW), Image.LW = epr_T2_LW(T2, Image.Mask); end
          
          Image.StartTime = safeget(s1.raw_info, 'StartTime', 0);
          Image.FinishTime = safeget(s1.raw_info, 'FinishTime', 0);
          Image.SourceFileName = s1.source_image;
          Image.Size = s1.mat_rec_info.rec.Size;
          
          if ~isempty(T2) && (add_out(LOAD_CC) || add_out(LOAD_PO2) || ...
              add_out(LOAD_RAWFIT))
            
            %             s1.mat_pO2_info.llw_zero_po2 = 12.4;
            % Homogeneous phantom code
            %             fake_amp_map = s1.mat_pO2_info.amp1mM *ones(size(Image.Amp));
            %             Image.pO2 = td_T2_PO2(T2, fake_amp_map, Image.Mask, s1.mat_pO2_info);
            
            % Copy all parameters from the given structure
            if exist('mat_pO2_info', 'var')
              s1.mat_pO2_info = check_mat_pO2_info(s1.mat_pO2_info, mat_pO2_info);
            else
              s1.mat_pO2_info = check_mat_pO2_info([], s1.mat_pO2_info);
            end
            
            % manual selection of parameters
            if ConstDialog
              s1.mat_pO2_info = EditConstDialog(s1.mat_pO2_info);
            elseif ~exist('mat_pO2_info', 'var')
              s1.mat_pO2_info.Q = safeget(s1.mat_pO2_info, 'Q', 15);
              s1.mat_pO2_info.Qcb = safeget(s1.mat_pO2_info, 'Qcb', s1.mat_pO2_info.Q);
            end
            
            Q_correction = sqrt(s1.mat_pO2_info.Qcb/s1.mat_pO2_info.Q);
            
            %  Image
            Image.pO2 = epr_T2_PO2(T2, Image.Amp, Image.Mask, s1.mat_pO2_info);
            Image.Amp = Image.Amp * Q_correction/ s1.mat_pO2_info.amp1mM;
            Image.pO2_info = s1.mat_pO2_info;
            Image.raw_info = s1.raw_info;
          end
        end
        
        if add_out(LOAD_RAW) || add_out(LOAD_RAW3D)
          [fpath, fname1, fext] = fileparts(fname);
          raw_file_name = fullfile(fpath, [fname1(2:end), fext]);
          if ~exist(raw_file_name, 'file')
            [raw_file_name, raw_path] = uigetfile({'*.mat', 'MATLAB File (*.mat)'; '*.*', 'All Files (*.*)'}, ...
              'Pick a 4D file', raw_file_name);
            if ~(isequal(raw_file_name,0) || isequal(raw_path,0))
              raw_file_name = fullfile(raw_path, raw_file_name);
            end
          end
          s2 = load(raw_file_name);
          if add_out(LOAD_RAW)
            Image.Raw  = s2.mat_recFXD;
            Image.Size = s2.mat_rec_info.rec.Size;
            Image.raw_info = s2.mat_info;
          elseif add_out(LOAD_RAW3D)
            Image.Raw  = epr_LoadSpatialImage(s2.mat_recFXD);
            Image.Size = s2.mat_rec_info.rec.Size;
          end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%                                   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%        %%%                %%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%      %%%%%              %%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%     %%%%%%             %%%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%    %%%%%%%           %%%%%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%   %%%%%%%%          %%%%%%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%       %%%%              %%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%       %%%%              %%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%       %%%%              %%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%       %%%%              %%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%       %%%%      %%      %%%%%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%    %%%%%%%%%%   %%   %%%%%%%%%%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%                                   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  case 'FitImage_v1.1'
    Image.rec_info = s1.rec_info;
    switch s1.fit_data.Algorithm
      case 'CW_spectral_fit_R2_XOVR_PHASE'
        add_out = load_options(AdditionalOutputsCW);
        
        % Copy all parameters from the given structure
        if exist('mat_pO2_info', 'var')
          s1.pO2_info = check_mat_pO2_info(s1.pO2_info, mat_pO2_info);
        else
          s1.pO2_info = check_mat_pO2_info([], s1.pO2_info);
        end
        
        % manual selection of parameters
        if ConstDialog
          s1.pO2_info = EditConstDialog(s1.pO2_info);
        elseif ~exist('mat_pO2_info', 'var')
          s1.pO2_info.Q = safeget(s1.pO2_info, 'Q', 15);
          s1.pO2_info.Qcb = safeget(s1.pO2_info, 'Qcb', s1.pO2_info.Q);
        end
        
        if add_out(LOAD_CC) || add_out(LOAD_RAWFIT) || add_out(LOAD_PO2), raw_int = LoadFitPars(s1.fit_data, {'Amp'} ); end
        if add_out(LOAD_PO2) || add_out(LOAD_LW), lw = LoadFitPars(s1.fit_data, {'LLW'} ); end
        if add_out(LOAD_LW), Image.LW = lw*1E3; end
        if add_out(LOAD_RAWFIT), Image.RAW_INT = raw_int; end
        %         if add_out(LOAD_ERROR), Image.Error = load_p(idx, s1.P_errs(2,:)); end
        %         if add_out(LOAD_PHASE), Image.PHASE = load_p(idx, s1.P(7,:)); end
        %         if add_out(LOAD_XOVR), Image.XOVER = load_p(idx, s1.P(1,:)); end
        Image.Size = s1.rec_info.rec.Size(1);
        Image.pO2_info = s1.pO2_info;
        Image.Mask = LoadFitPars(s1.fit_data, {'Mask'} );
        if add_out(LOAD_CC) || add_out(LOAD_PO2)
          raw_int = raw_int*safeget(Image.pO2_info, 'ampHH', 1)*safeget(Image.pO2_info, 'ampSS', 1);
          
          % quality factor correction
          Q_correction = sqrt(Image.pO2_info.Qcb/Image.pO2_info.Q);
          Image.Amp = raw_int * Q_correction/ Image.pO2_info.amp1mM;
          Image.Mask = Image.Mask & Image.Amp > 0;
        end
        if add_out(LOAD_PO2)
          Image.pO2 = epr_LLW_PO2(lw*1E3, raw_int, Image.Mask, Image.pO2_info);
        end
        if add_out(LOAD_XOVR), Image.XOVER = LoadFitPars(s1.fit_data, {'xover'} ); end
        if add_out(LOAD_PHASE), Image.PHASE = LoadFitPars(s1.fit_data, {'phase'} ); end
        if add_out(LOAD_ERROR), Image.Error = LoadFitPars(s1.fit_data, {'error'} ); end
        % raw data
        Image.raw_info = s1.raw_info;
        Image.rec_info = s1.rec_info;
        if add_out(LOAD_RAW) || add_out(LOAD_RAW3D)
          [fpath, fname1, fext] = fileparts(fname);
          raw_file_name = fullfile(fpath, [fname1(2:end), fext]);
          if ~exist(raw_file_name, 'file')
            [raw_file_name, raw_path] = uigetfile({'*.mat', 'MATLAB File (*.mat)'; '*.*', 'All Files (*.*)'}, ...
              'Pick a 4D file', raw_file_name);
            if ~(isequal(raw_file_name,0) || isequal(raw_path,0))
              raw_file_name = fullfile(raw_path, raw_file_name);
            end
          end
          s2 = load(raw_file_name);
          if add_out(LOAD_RAW)
            Image.Raw  = s2.mat_recFXD;
          else
            Image.Raw  = epr_LoadSpatialImage(s2.mat_recFXD);
          end
          Image.raw_info = s2.raw_info;
        end
      case 'T2_ExpDecay_No_Offset'
        add_out = load_options(AdditionalOutputsESE);
        
        % Copy all parameters from the given structure
        if exist('mat_pO2_info', 'var')
          s1.pO2_info = check_mat_pO2_info(s1.pO2_info, mat_pO2_info);
        else
          s1.pO2_info = check_mat_pO2_info([], s1.pO2_info);
        end
        
        if add_out(LOAD_CC) || add_out(LOAD_PO2) || add_out(LOAD_ERROR) || ...
            add_out(LOAD_RAWFIT) || add_out(LOAD_T2)
          if add_out(LOAD_ERROR)
            [Image.Amp, T2, Image.Mask, Image.Error] = LoadFitPars(safeget(s1, 'fit_data', ''),...
              {'Amp','T2','Mask','Error'});
          else
            [Image.Amp, T2, Image.Mask] = LoadFitPars(safeget(s1, 'fit_data', ''));
            Image.Error = [];
          end
          if add_out(LOAD_T2), Image.T2 = T2; end
          if add_out(LOAD_LW), Image.LW = epr_T2_LW(T2, Image.Mask); end
          
          Image.StartTime = safeget(s1.raw_info, 'StartTime', 0);
          Image.FinishTime = safeget(s1.raw_info, 'FinishTime', 0);
          Image.SourceFileName = s1.source_image;
          Image.Size = s1.rec_info.rec.Size;
          
          if ~isempty(T2) && (add_out(LOAD_CC) || add_out(LOAD_PO2) || ...
              add_out(LOAD_RAWFIT))
            
            %             s1.mat_pO2_info.llw_zero_po2 = 12.4;
            % Homogeneous phantom code
            %             fake_amp_map = s1.mat_pO2_info.amp1mM *ones(size(Image.Amp));
            %             Image.pO2 = td_T2_PO2(T2, fake_amp_map, Image.Mask, s1.mat_pO2_info);
            
            % Copy all parameters from the given structure
            if exist('mat_pO2_info', 'var')
              s1.pO2_info = check_mat_pO2_info(s1.pO2_info, mat_pO2_info);
            else
              s1.pO2_info = check_mat_pO2_info([], s1.pO2_info);
            end
            
            % manual selection of parameters
            if ConstDialog
              s1.pO2_info = EditConstDialog(s1.pO2_info);
            elseif ~exist('mat_pO2_info', 'var')
              s1.pO2_info.Q = safeget(s1.pO2_info, 'Q', 15);
              s1.pO2_info.Qcb = safeget(s1.pO2_info, 'Qcb', s1.pO2_info.Q);
            end
            
            Q_correction = sqrt(s1.pO2_info.Qcb/s1.pO2_info.Q);
            
            %  Image
            Image.pO2 = epr_T2_PO2(T2, Image.Amp, Image.Mask, s1.pO2_info);
            Image.Amp = Image.Amp * Q_correction/ s1.pO2_info.amp1mM;
            Image.pO2_info = s1.pO2_info;
            Image.raw_info = s1.raw_info;
          end
        end
        
        if add_out(LOAD_RAW) || add_out(LOAD_RAW3D)
          [fpath, fname1, fext] = fileparts(fname);
          raw_file_name = fullfile(fpath, [fname1(2:end), fext]);
          if ~exist(raw_file_name, 'file')
            [raw_file_name, raw_path] = uigetfile({'*.mat', 'MATLAB File (*.mat)'; '*.*', 'All Files (*.*)'}, ...
              'Pick a 4D file', raw_file_name);
            if ~(isequal(raw_file_name,0) || isequal(raw_path,0))
              raw_file_name = fullfile(raw_path, raw_file_name);
            end
          end
          s2 = load(raw_file_name);
          if isfield(s2, 'mat_rec_info') && ~isfield(s2, 'rec_info')
            s2.rec_info = s2.mat_rec_info;
          end
          if add_out(LOAD_RAW)
            Image.Raw  = s2.mat_recFXD;
            Image.Size = s2.rec_info.rec.Size;
            Image.raw_info = s2.raw_info;
          elseif add_out(LOAD_RAW3D)
            Image.Raw  = epr_LoadSpatialImage(s2.mat_recFXD);
            Image.Size = s2.rec_info.rec.Size;
          end
        end
      case 'T1_InvRecovery_3Par'
        add_out = load_options(AdditionalOutputsESE);
        [Image.Amp, T1, Image.Mask] = LoadFitPars(safeget(s1, 'fit_data', ''),{'Amp','T1','Mask'});
        Image.raw_info = s1.raw_info;
        if add_out(LOAD_T1), Image.T1 = T1; end
        if add_out(LOAD_R1), Image.R1 = epr_T2_LW(T1,Image.Mask); end
       
        % Copy all parameters from the given structure
        if exist('mat_pO2_info', 'var')
          s1.pO2_info = check_mat_pO2_info(s1.pO2_info, mat_pO2_info);
        else
          s1.pO2_info = check_mat_pO2_info([], s1.pO2_info);
        end
        
        % manual selection of parameters
        if ConstDialog
          s1.pO2_info = EditConstDialog(s1.pO2_info);
        elseif ~exist('mat_pO2_info', 'var')
          s1.pO2_info.Q = safeget(s1.pO2_info, 'Q', 15);
          s1.pO2_info.Qcb = safeget(s1.pO2_info, 'Qcb', s1.pO2_info.Q);
        end
        
        Image.StartTime = safeget(s1.raw_info, 'StartTime', 0);
        Image.FinishTime = safeget(s1.raw_info, 'FinishTime', 0);
        Image.SourceFileName = safeget(s1, 'name_com', '');
        
        tau_correction = exp(2*630e-3*median(1./T1(Image.Mask(:))));
        %         Q_correction = sqrt(s1.pO2_info.Qcb/s1.pO2_info.Q);
        Q_correction = 1;
        Image.Amp = Image.Amp * Q_correction .* tau_correction;
        Image.pO2 = epr_T2_PO2(T1, Image.Amp, Image.Mask, s1.pO2_info);
        Image.Amp = Image.Amp / s1.pO2_info.amp1mM;
        Image.SourceFileName = s1.source_image;
        Image.Size = s1.rec_info.rec.Size;
        Image.pO2_info = s1.pO2_info;
        if add_out(LOAD_LW), Image.LW = epr_T2_LW(T1, Image.Mask); end
        if add_out(LOAD_ERR_T2|LOAD_ERR_T1)
          Image.Error_T1 = LoadFitPars(safeget(s1, 'fit_data', ''), {'ERROR_T1'});
        end
        if add_out(LOAD_RAW) || add_out(LOAD_RAW3D)
          [fpath, fname1, fext] = fileparts(fname);
          raw_file_name = fullfile(fpath, [fname1(2:end), fext]);
          if ~exist(raw_file_name, 'file')
            [raw_file_name, raw_path] = uigetfile({'*.mat', 'MATLAB File (*.mat)'; '*.*', 'All Files (*.*)'}, ...
              'Pick a 4D file', raw_file_name);
            if ~(isequal(raw_file_name,0) || isequal(raw_path,0))
              raw_file_name = fullfile(raw_path, raw_file_name);
            end
          end
          s2 = load(raw_file_name);
          if add_out(LOAD_RAW)
            Image.Raw  = s2.mat_recFXD;
            Image.Size = s2.rec_info.rec.Size;
            Image.raw_info = s2.raw_info;
          elseif add_out(LOAD_RAW3D)
            Image.Raw  = epr_LoadSpatialImage(s2.mat_recFXD);
            Image.Size = s2.rec_info.rec.Size;
          end
        end
      case 'T2T1_InvRecovery_3Par'
        add_out = load_options(AdditionalOutputsESE);
        [Image.Amp, T1, Image.Mask] = LoadFitPars(safeget(s1, 'fit_data', ''),{'Amp','T1','Mask'});
        Image.raw_info = s1.raw_info;
        if add_out(LOAD_T1), Image.T1 = T1; end
        
        % Copy all parameters from the given structure
        if exist('mat_pO2_info', 'var')
          s1.pO2_info = check_mat_pO2_info(s1.pO2_info, mat_pO2_info);
        else
          s1.pO2_info = check_mat_pO2_info([], s1.pO2_info);
        end
        
        % manual selection of parameters
        if ConstDialog
          s1.pO2_info = EditConstDialog(s1.pO2_info);
        elseif ~exist('mat_pO2_info', 'var')
          s1.pO2_info.Q = safeget(s1.pO2_info, 'Q', 15);
          s1.pO2_info.Qcb = safeget(s1.pO2_info, 'Qcb', s1.pO2_info.Q);
        end
        
        Image.StartTime = safeget(s1.raw_info, 'StartTime', 0);
        Image.FinishTime = safeget(s1.raw_info, 'FinishTime', 0);
        Image.SourceFileName = safeget(s1, 'name_com', '');
        
        %         Q_correction = sqrt(s1.pO2_info.Qcb/s1.pO2_info.Q);
        Q_correction = 1;
        Image.Amp = Image.Amp * Q_correction;
        Image.pO2 = epr_T2_PO2(T1, Image.Amp, Image.Mask, s1.pO2_info);
        Image.Amp = Image.Amp / s1.pO2_info.amp1mM;
        Image.SourceFileName = s1.source_image;
        Image.Size = s1.rec_info.rec.Size;
        Image.pO2_info = s1.pO2_info;
        if add_out(LOAD_LW|LOAD_T2), 
          T2 = LoadFitPars(safeget(s1, 'fit_data', ''), {'T2'});
          if add_out(LOAD_T2), Image.T2 = T2; end
          if add_out(LOAD_LW), Image.LW = epr_T2_LW(T2, Image.Mask); end
        end
        if add_out(LOAD_ERR_T2)
          Image.Error_T2 = LoadFitPars(safeget(s1, 'fit_data', ''), {'ERROR_T2'});
        end
        if add_out(LOAD_ERR_T1)
          Image.Error_T1 = LoadFitPars(safeget(s1, 'fit_data', ''), {'ERROR_T1'});
        end
        if add_out(LOAD_RAW) || add_out(LOAD_RAW3D)
          [fpath, fname1, fext] = fileparts(fname);
          raw_file_name = fullfile(fpath, [fname1(2:end), fext]);
          if ~exist(raw_file_name, 'file')
            [raw_file_name, raw_path] = uigetfile({'*.mat', 'MATLAB File (*.mat)'; '*.*', 'All Files (*.*)'}, ...
              'Pick a 4D file', raw_file_name);
            if ~(isequal(raw_file_name,0) || isequal(raw_path,0))
              raw_file_name = fullfile(raw_path, raw_file_name);
            end
          end
          s2 = load(raw_file_name);
          if add_out(LOAD_RAW)
            Image.Raw  = s2.mat_recFXD;
            Image.Size = s2.rec_info.rec.Size;
            Image.raw_info = s2.raw_info;
          elseif add_out(LOAD_RAW3D)
            Image.Raw  = epr_LoadSpatialImage(s2.mat_recFXD);
            Image.Size = s2.rec_info.rec.Size;
          end
        end   
      case 'Rabi_sin_2Par'
         add_out = load_options(AdditionalOutputsESE);
        [Image.Amp, T1, Image.Mask] = LoadFitPars(safeget(s1, 'fit_data', ''),{'Amp','tp','Mask'});
        Image.raw_info = s1.raw_info;
        if add_out(LOAD_T1), Image.T1 = T1; end
        
        % Copy all parameters from the given structure
        if exist('mat_pO2_info', 'var')
          s1.pO2_info = check_mat_pO2_info(s1.pO2_info, mat_pO2_info);
        else
          s1.pO2_info = check_mat_pO2_info([], s1.pO2_info);
        end

        s1.pO2_info.Q = safeget(s1.pO2_info, 'Q', 15);
        s1.pO2_info.Qcb = safeget(s1.pO2_info, 'Qcb', s1.pO2_info.Q);

        
        Image.StartTime = safeget(s1.raw_info, 'StartTime', 0);
        Image.FinishTime = safeget(s1.raw_info, 'FinishTime', 0);
        Image.SourceFileName = safeget(s1, 'name_com', '');
        
        %         Q_correction = sqrt(s1.pO2_info.Qcb/s1.pO2_info.Q);
        Q_correction = 1;
        Image.Amp = Image.Amp * Q_correction;
        Image.pO2 = T1;
        Image.Amp = Image.Amp / s1.pO2_info.amp1mM;
        Image.SourceFileName = s1.source_image;
        Image.Size = s1.rec_info.rec.Size;
        Image.pO2_info = s1.pO2_info;

        if add_out(LOAD_RAW) || add_out(LOAD_RAW3D)
          [fpath, fname1, fext] = fileparts(fname);
          raw_file_name = fullfile(fpath, [fname1(2:end), fext]);
          if ~exist(raw_file_name, 'file')
            [raw_file_name, raw_path] = uigetfile({'*.mat', 'MATLAB File (*.mat)'; '*.*', 'All Files (*.*)'}, ...
              'Pick a 4D file', raw_file_name);
            if ~(isequal(raw_file_name,0) || isequal(raw_path,0))
              raw_file_name = fullfile(raw_path, raw_file_name);
            end
          end
          s2 = load(raw_file_name);
          if add_out(LOAD_RAW)
            Image.Raw  = s2.mat_recFXD;
            Image.Size = s2.rec_info.rec.Size;
            Image.raw_info = s2.raw_info;
          elseif add_out(LOAD_RAW3D)
            Image.Raw  = epr_LoadSpatialImage(s2.mat_recFXD);
            Image.Size = s2.rec_info.rec.Size;
          end
        end          
    end
  case 'ImageMask_v1.0'
    Image.Raw  = s1.Mask;
    Image.Size = size(s1.Mask);
  otherwise
    disp(file_type);
end

if isSingleOutput
  switch upper(AdditionalOutputsESE{1})
    case 'ERROR', Image = Image.Error;
    case 'SNR', Image = Image.SNR;
    case 'LW',  Image = Image.LW;
    case 'T2',  Image = Image.T2;
    case 'RAWFIT',  Image = Image.RAW_INT;
    case 'PO2',  Image = Image.pO2;
    case 'CC',  Image = Image.Amp;
    case 'RAW',  Image = Image.Raw;
    case 'RAW3D', Image = Image.Raw;
    case 'PHASE', Image = Image.PHASE;
    case 'XOVER', Image = Image.XOVER;
    case 'ERROR_T2', Image = Image.Error_T2;
  end
else
  % Recovery from an empty data
  if isempty(Image.Size)
    switch file_type
      case 'Image_v1.0'
        if isfield(s1, 'mat_recFXD') && ~isempty(s1.mat_recFXD)
          if strcmp(LoadMode, '3D') || ...
              strcmp(questdlg('Do you want to use spatial image instead of parametric ?', ...
              'Question', 'Yes', 'No', 'Yes'), 'Yes')
            Image.Amp  = td_LoadSpatialImage(s1.mat_recFXD);
            Image.Size = s1.mat_rec_info.rec.Size;
            Image.pO2  = rand(size(Image.Amp));
            % use mask from the image
            if isfield(s1, 'Mask') && ~isempty(s1.Mask)
              Image.Mask = s1.Mask;
            else
              Image.Mask = true(size(Image.Amp));
            end
          end
        end
    end
  end
end

% --------------------------------------------------------------------
function out_data = load_p(in_mask, in_data)
out_data = zeros(size(in_mask)); out_data(in_mask) = in_data;
out_data = permute(out_data, [3,2,1]);

% --------------------------------------------------------------------
function Concentration = Int2ConcentrationCW(intMap, pars_out, calb)
intMap(isnan(intMap)) = 0;

% Sensitivity correction
intMap=intMap*pars_out(21);

% nbins correction
nbins=size(intMap,1);
intMap=(nbins/64)^3*intMap; % convert intMap to 64-bin intensity

% image size correction
intMap=(pars_out(8)/(3*sqrt(2)))^3*intMap; % convert intMap to 3*sqrt(2) cm intensity

% mod amp correction
modampG=pars_out(16)*calb.ModCalImage;
intMap=intMap / (1 + calb.ModBroadn_1 * (modampG-calb.ModBroadn_0)/calb.ModBroadn_0);

% quality factor correction
Q_correction = sqrt(calb.Qcb/calb.Q);
intMap = intMap * Q_correction;

% convert readings into mM
Concentration = intMap / calb.amp1mM;

% --------------------------------------------------------------------
function mat_pO2_info = EditConstDialog(mat_pO2_info)

prompt = {'Enter LLW(0 torr) [mG]:', ...
  'Enter pO2 coefficient [torr/mG]:', ...
  'Enter amplitude coefficient [mG/mM]:',...
  'Enter AVERAGE amplitude coefficient [mG/mM]:',...
  'Enter 1 mM signal [a.u.]:',...
  'Quality factor during THE IMAGE', ...
  'Quality factor during CALIBRATION'};
mat_pO2_info.mG_per_mM = safeget(mat_pO2_info, 'mG_per_mM', 2.32);
mat_pO2_info.MDNmG_per_mM = safeget(mat_pO2_info, 'MDNmG_per_mM', 0);
mat_pO2_info.LLW_zero_po2 = safeget(mat_pO2_info, 'LLW_zero_po2', 12.4);
mat_pO2_info.amp1mM = safeget(mat_pO2_info, 'amp1mM', 1);
mat_pO2_info.Q = safeget(mat_pO2_info, 'Q', 15);
mat_pO2_info.Qcb = safeget(mat_pO2_info, 'Qcb', mat_pO2_info.Q);
mat_pO2_info.Torr_per_mGauss = safeget(mat_pO2_info, 'Torr_per_mGauss', 1.84);
def = {num2str(mat_pO2_info.LLW_zero_po2),...
  num2str(mat_pO2_info.Torr_per_mGauss),...
  num2str(mat_pO2_info.mG_per_mM), ...
  num2str(mat_pO2_info.MDNmG_per_mM), ...
  num2str(mat_pO2_info.amp1mM), ...
  num2str(mat_pO2_info.Q), ...
  num2str(mat_pO2_info.Qcb), ...
  };
answer = inputdlg(prompt,'Constants',1,def);
if ~isempty(answer)
  try mat_pO2_info.LLW_zero_po2 = eval(answer{1}); catch err, disp(err); end
  try mat_pO2_info.Torr_per_mGauss = eval(answer{2}); catch err, disp(err); end
  try mat_pO2_info.mG_per_mM = eval(answer{3}); catch err, disp(err); end
  try mat_pO2_info.MDNmG_per_mM = eval(answer{4}); catch err, disp(err); end
  try mat_pO2_info.amp1mM = eval(answer{5}); catch err, disp(err); end
  try mat_pO2_info.Q = eval(answer{6}); catch err, disp(err); end
  try mat_pO2_info.Qcb = eval(answer{7}); catch err, disp(err); end
end

% --------------------------------------------------------------------
function [add_out, field_names] = load_options(pars)
LOAD_PO2 = 1; LOAD_CC = 2; LOAD_RAW = 3;
LOAD_ERROR = 4; LOAD_SNR = 5; LOAD_LW = 6; LOAD_T2 = 7; LOAD_T1 = 8;
LOAD_RAWFIT = 9; LOAD_RAW3D = 10; LOAD_PHASE = 11;
LOAD_XOVR = 12; LOAD_PRJ = 13; LOAD_ERR_T2 = 14; LOAD_ERR_T1 = 15;
LOAD_R1 = 16; LOAD_LAST = 17;
add_out = false(LOAD_LAST,1);

field_names = {};

for ii=1:length(pars)
  switch upper(pars{ii})
    case 'ERROR', add_out(LOAD_ERROR) = true;
    case 'SNR', add_out(LOAD_SNR) = true;
    case {'LW','LLW'},  add_out(LOAD_LW) = true;
    case 'T1',  add_out(LOAD_T1) = true;
    case 'T2',  add_out(LOAD_T2) = true;
    case 'R1',  add_out(LOAD_R1) = true;      
    case 'RAWFIT',  add_out(LOAD_RAWFIT) = true;
    case 'PO2',  add_out(LOAD_PO2) = true;
    case 'CC',  add_out(LOAD_CC) = true;
    case 'RAW',  add_out(LOAD_RAW) = true;
    case 'RAW3D', add_out(LOAD_RAW3D) = true;
    case 'PHASE', add_out(LOAD_PHASE) = true;
    case 'XOVER', add_out(LOAD_XOVR) = true;
    case 'PRJ', add_out(LOAD_PRJ) = true;
    case 'ERROR_T2', add_out(LOAD_ERR_T2) = true;
    case 'ERROR_T1', add_out(LOAD_ERR_T1) = true;
  end
end

% --------------------------------------------------------------------
function mat_pO2_info = check_mat_pO2_info(mat_pO2_info, add_mat_pO2_info)

field_list     = {'Torr_per_mGauss', 'LLW_zero_po2', 'mG_per_mM', 'MDNmG_per_mM', 'amp1mM', 'ampHH', 'Q', 'Qcb'};
default_values = [1.84, 12.4, 0, 0, 1, 1, 15, safeget(mat_pO2_info, 'Q', 15)];

for ii=1:length(field_list)
  mat_pO2_info.(field_list{ii}) = ...
    safeget(add_mat_pO2_info, field_list{ii}, ...
    safeget(mat_pO2_info, field_list{ii}, default_values(ii)));
end
