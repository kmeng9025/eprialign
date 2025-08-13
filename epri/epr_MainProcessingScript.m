function FinalImage = epr_MainProcessingScript(data, file_name, type)
FinalImage = [];

isoptions = evalin('base', 'exist(''options'', ''var'')');
if isoptions
  options = evalin('base', 'options');
else
  options = [];
end

% date_path = epr_DateFromPath(file_name);
output_path = fileparts(file_name);

td_CreateDirectory(file_name);

switch type
  case 'ESET2_RAW'
    close all
    % get profile
    is_profile = evalin('base', 'exist(''LastProfile'', ''var'')');
    fields = epr_LoadScenario('PulseRecon.scn', 'PulseSOP Image Reconstruction.par');
    load_opt.FBP = fields.fbp;
    load_opt.Modality = 'PULSEFBP';
    load_opt.Sequence  = '2pECHO';
    if is_profile
      fields.fft.profile_file = evalin('base', 'LastProfile');
    end
    
    [FinalImage.mat,FinalImage.mat_bl,FinalImage.raw_info,load_opt]=epr_ReadPulseImageFile(uint8(data(:)), load_opt);
    fields.fbp = load_opt.FBP;
    fields.prc.save_data = 'yes';
    FinalImage.raw_info.FileName = file_name;
    
    if isfield(load_opt.FBP, 'MB0') && ~isempty(load_opt.FBP.MB0)
      all_fields = fieldnames(load_opt.FBP.MB0);
      for ii=1:length(all_fields)
        fields.MB0.(all_fields{ii}) = load_opt.FBP.MB0.(all_fields{ii});
      end
    else
      fields.MB0 = safeget(fields, 'MBpars', []);
    end

    FinalImage = ese_fbp(FinalImage, '', output_path, fields);
    disp(FinalImage)
    
    ImageBrowserGUI(FinalImage)
    %*******************************************************
    %*******************************************************
    %*******************************************************
    %*******************************************************
    %*******************************************************
    
  case 'ESE3pT1_RAW'
    close all
    % get profile
    is_profile = evalin('base', 'exist(''LastProfile'', ''var'')');
    fields = epr_LoadScenario('PulseRecon.scn', 'T1 3p Image Reconstruction.par');
    load_opt.FBP = fields.fbp;
    load_opt.Modality = 'PULSEFBP';
    load_opt.Sequence  = '3pT1';
    if is_profile
      fields.fft.profile_file = evalin('base', 'LastProfile');
    end
    
    [FinalImage.mat,FinalImage.mat_bl,FinalImage.raw_info, load_opt]=epr_ReadPulseImageFile(uint8(data(:)), load_opt);
    fields.prc.save_data = 'yes';
    FinalImage.raw_info.FileName = file_name;
    
    FinalImage = ese_fbp_3pT1(FinalImage, '', output_path, fields);
    disp(FinalImage)
    
    ImageBrowserGUI(FinalImage)
    
    %*******************************************************
    %*******************************************************
    %*******************************************************
    %*******************************************************
    %*******************************************************
    
  case 'ESET1T2_RAW'
    close all
    % get profile
    is_profile = evalin('base', 'exist(''LastProfile'', ''var'')');
    fields = epr_LoadScenario('PulseRecon.scn', 'Pulse T1 inversion recovery and T2 combined.par');
    load_opt.Modality = 'PULSEFBP';
    load_opt.Sequence  = 'ESEInvRec';
    if is_profile
      fields.fft.profile_file = evalin('base', 'LastProfile');
    end
    
    [FinalImage.mat,FinalImage.mat_bl,FinalImage.raw_info, load_opt]=epr_ReadPulseImageFile(uint8(data(:)), load_opt);
    fields.fbp = load_opt.FBP;
    fields.prc.save_data = 'yes';
    FinalImage.raw_info.FileName = file_name;
    
    if isfield(load_opt.FBP, 'MB0') && ~isempty(load_opt.FBP.MB0)
      all_fields = fieldnames(load_opt.FBP.MB0);
      for ii=1:length(all_fields)
        fields.MB0.(all_fields{ii}) = load_opt.FBP.MB0.(all_fields{ii});
      end
    else
      fields.MB0 = safeget(fields, 'MBpars', []);
    end
    
    FinalImage = ese_fbp_InvRec(FinalImage, '', output_path, fields);
    disp(FinalImage)
    
    ImageBrowserGUI(FinalImage)
  case 'ESEInvRec_RAW'
    close all
    % get profile
    is_profile = evalin('base', 'exist(''LastProfile'', ''var'')');
    fields = epr_LoadScenario('PulseRecon.scn', 'Pulse T1 inversion recovery.par');
    load_opt.Modality = 'PULSEFBP';
    load_opt.Sequence  = 'ESEInvRec';
    if is_profile
      fields.fft.profile_file = evalin('base', 'LastProfile');
    end
    
    [FinalImage.mat,FinalImage.mat_bl,FinalImage.raw_info, load_opt]=epr_ReadPulseImageFile(uint8(data(:)), load_opt);
    fields.fbp = load_opt.FBP;
    fields.prc.save_data = 'yes';
    FinalImage.raw_info.FileName = file_name;
    
    if isfield(load_opt.FBP, 'MB0') && ~isempty(load_opt.FBP.MB0)
      all_fields = fieldnames(load_opt.FBP.MB0);
      for ii=1:length(all_fields)
        fields.MB0.(all_fields{ii}) = load_opt.FBP.MB0.(all_fields{ii});
      end
    else
      fields.MB0 = safeget(fields, 'MBpars', []);
    end
    
    FinalImage = ese_fbp_InvRec(FinalImage, '', output_path, fields);
    disp(FinalImage)
    
    ImageBrowserGUI(FinalImage)

    %*******************************************************
    %*******************************************************
    %*******************************************************
    %*******************************************************
    %*******************************************************
  case 'ESE3D_RAW'
    close all
    is_profile = evalin('base', 'exist(''LastProfile'', ''var'')');
    fields = epr_LoadScenario('PulseRecon.scn', 'Pulse Fiducials Image Reconstruction.par');
    load_opt.FBP = fields.fbp;
    load_opt.Modality = 'PULSEFBP';
    load_opt.Sequence  = '2pECHO';
    if is_profile
      fields.fft.profile_file = evalin('base', 'LastProfile');
    end
    
    [FinalImage.mat,FinalImage.mat_bl,FinalImage.raw_info, load_opt]=epr_ReadPulseImageFile(uint8(data(:)), load_opt);
    fields.prc.save_data = 'yes';
    FinalImage.raw_info.FileName = file_name;

    if ~isempty(fields.fbp)
      fields.fbp=load_opt.FBP;
    end
    
    if isfield(load_opt.FBP, 'MB0') && ~isempty(load_opt.FBP.MB0)
      all_fields = fieldnames(load_opt.FBP.MB0);
      for ii=1:length(all_fields)
        fields.MB0.(all_fields{ii}) = load_opt.FBP.MB0.(all_fields{ii});
      end
    else
      fields.MB0 = safeget(fields, 'MBpars', []);
    end
    
    FinalImage = ese_fbp(FinalImage, '', output_path, fields);
    FinalImage.Raw(FinalImage.Raw < 0) = 0;
    disp(FinalImage)
    
    ImageBrowserGUI(FinalImage)
    %*******************************************************
    %*******************************************************
    %*******************************************************
    %*******************************************************
    %*******************************************************
    
  case 'cprofile'
    close all
    [FinalImage.raw_info,FinalImage.mat,FinalImage.dsc]=kv_smtdmsread(uint8(data(:)));
    FinalImage.raw_info.FileName = file_name;
    
    FinalImage = ese_cprofile(FinalImage, file_name);
    assignin('base', 'LastProfile', FinalImage.FileName);
    
    freq = real(FinalImage.frequency(~FinalImage.Exclude));
    amp  = real(FinalImage.amplitude(~FinalImage.Exclude));
    
    [freq, freq_idx] = sort(freq);
    amp = amp(freq_idx);
    
    figure(1); clf;
    plot(freq, amp); hold on;
    title(sprintf('cavity profile: %s', epr_ShortFileName(FinalImage.FileName, 45)));
    
    [mm, m_idx] = max(amp);
    off3dBmin = find(amp(1:m_idx) > mm * 0.5, 1, 'first');
    off3dBmax = find(amp(m_idx:end) > mm * 0.5, 1, 'last');
    p1=polyfit(freq(off3dBmin-1:off3dBmin+1),amp(off3dBmin-1:off3dBmin+1),1);
    p2=polyfit(freq(m_idx+off3dBmax-2:m_idx+off3dBmax),amp(m_idx+off3dBmax-2:m_idx+off3dBmax),1);
    f1 = (mm*0.5 - p1(2))/p1(1);
    f2 = (mm*0.5 - p2(2))/p2(1);
    text(freq(m_idx), mm*.54, sprintf('%4.2f',f2-f1),'HorizontalAlignment','center')
    plot([f1, f2], mm*.5*[1,1],'r')
    
    axis([-inf, inf, 0, inf]);
    xlabel('Frequency [MHz]');
    ylabel('Amplitude [a.u.]');
  case 'T1'
    [raw_info,curve,FinalImage.dsc]=kv_smtdmsread(uint8(data(:)));
    if safeget(options, 'isDetailedLog', false)
      assignin('base', 'dsc', FinalImage.dsc);
    end
    T1 = raw_info.x*1E6;
    curve = curve(:, 1) - curve(:, 2);
    
    figure(1003); subplot(2,2,1);
    [fit_amp, fit_t1, fit_inv, fit_err_mask, fit_error] = fit_recovery_3par( real(curve'), T1);
    
    plot(T1, real(curve), 'o', ...
      T1, fit_amp - fit_inv*exp(-T1/fit_t1), '-'); xlabel('T1 [us]')
    
    text(0.75, .1, sprintf('T1 = %5.2f us', fit_t1), 'Units', 'Normalized')
    
    AddTimeline(fit_t1, true, options);
  case 'T2'
    [raw_info,curve,FinalImage.dsc]=kv_smtdmsread(uint8(data(:)));
    if safeget(options, 'isDetailedLog', false)
      assignin('base', 'dsc', FinalImage.dsc);
    end
    T2 = 2*raw_info.x*1E6;
    curve = curve(:, 1) - curve(:, 2);
    
    figure(1003); subplot(2,2,2);
    [fit_amp, fit_t2, fit_err_mask, fit_error] = fit_exp_no_offset( real(curve'), T2);
    
    plot(T2, real(curve), 'o', ...
      T2, fit_amp*exp(-T2/fit_t2), '-'); xlabel('T2 [us]')
    
    text(0.75, .9, sprintf('T2 = %5.2f us', fit_t2), 'Units', 'Normalized')
    
    AddTimeline(fit_t2, false, options);
  otherwise
    tdms = epr_LoadTDMS(uint8(data(:)));
    if safeget(options, 'isDetailedLog', false)
      assignin('base', 'tdms', tdms);
    end
    FinalImage = tdms;
    
    tdms.axis

    tt = 1:length(tdms.streams.Re.data{1});
    figure(1010); clf; hold on
    plot(tt, tdms.streams.Re.data{1}, 'b.-');
    plot(tt, tdms.streams.Im.data{1}, 'r.-');
    legend({'Re', 'Im'})
    disp(output_path)

    
    [fpath, ff] = fileparts(file_name);
    
    % saving
    save(fullfile(fpath, [ff, '.mat']), '-struct', 'FinalImage');
end

% --------------------------------------------------------------------
function hh = OpenGUI(handles)
set(0, 'ShowHiddenHandles', 'on');
h = findobj('Tag', 'cbConnect12321');
set(0, 'ShowHiddenHandles', 'off');
hh = [];
for ii=1:length(h),
  if h(ii) ~= handles.cbConnect12321 && get(h(ii), 'Value')
    hh(end+1) = h(ii);
  end;
end

% --------------------------------------------------------------------
function AddTimeline(val, isT1, options)
isT2HistoryActive = safeget(options, 'isT2HistoryActive', false);
isT1HistoryActive = safeget(options, 'isT1HistoryActive', false);
isT2history = evalin('base', 'exist(''T2history'',''var'')');
if ~isT2history, T2history = []; else T2history = evalin('base', 'T2history'); end
isT1history = evalin('base', 'exist(''T1history'',''var'')');
if ~isT1history, T1history = []; else T1history = evalin('base', 'T1history'); end

if isT2HistoryActive
  if ~isT1
    T2history{end+1} = struct('time', now, 'T2', val);
    assignin('base', 'T2history', T2history);
  end
else
  T2history = {};
end
if isT1HistoryActive
  if isT1
    T1history{end+1} = struct('time', now, 'T1', val);
    assignin('base', 'T1history', T1history);
  end
else
  T1history = {};
end
if isT2HistoryActive || isT1HistoryActive
  time_line_T2 = zeros(length(T2history), 1);
  time_T2 = zeros(length(T2history), 1);
  for ii=1:length(T2history)
    time_line_T2(ii) = T2history{ii}.time;
    time_T2(ii) = T2history{ii}.T2;
  end
  time_line_T1 = zeros(length(T1history), 1);
  time_T1 = zeros(length(T1history), 1);
  for ii=1:length(T1history)
    time_line_T1(ii) = T1history{ii}.time;
    time_T1(ii) = T1history{ii}.T1;
  end
  n_stat = 5;
  time_now = now;
   
  if safeget(options, 'isSaveT1T2', false)
    ouput_data_T2 = [24*60*(time_now-time_line_T2), time_T2];
    ouput_data_T1 = [24*60*(time_now-time_line_T1), time_T1];
    save('C:\ProcessingServerSharp\Data\ouput_data_T2.dat', 'ouput_data_T2', '-ascii')
    save('C:\ProcessingServerSharp\Data\ouput_data_T1.dat', 'ouput_data_T1', '-ascii')
  end
  
  figure(1003); h = subplot(2,1,2); cla(h); hold on;
  plot(24*60*(time_now-time_line_T2), time_T2, 'o', 24*60*(now-time_line_T1), time_T1, 'o');
  xlabel(''); ylabel('T1/T2 [us]');
  if length(time_T2) > n_stat
    time_line_T2 = time_line_T2(end-5:end); 
    time_T2 = time_T2(end-5:end); 
    mean_time_T2 = mean(time_T2);
    std_time_T2 = std(time_T2);
    plot(24*60*(time_now - [time_line_T2(1), time_line_T2(end)]), mean_time_T2*[1,1], 'r');
    plot(24*60*(time_now - [1,1]*mean(time_line_T2)), mean_time_T2 + [1, -1]*std_time_T2, 'r');
    text(.05, 0.5, sprintf('mn_{T2}=%5.3f(+-%5.3f)us',mean_time_T2, std_time_T2),'Units','normalized')
  end
  if length(time_T1) > n_stat
    time_line_T1 = time_line_T1(end-5:end); 
    time_T1 = time_T1(end-5:end); 
    mean_time_T1 = mean(time_T1);
    std_time_T1 = std(time_T1);
    plot(24*60*(time_now - [time_line_T1(1), time_line_T1(end)]), mean_time_T1*[1,1], 'r');
    plot(24*60*(time_now - [1,1]*mean(time_line_T1)), mean_time_T1 + [1, -1]*std_time_T1, 'r');
    text(.5, 0.5, sprintf('mn_{T1}=%5.3f(+-%5.3f)us',mean_time_T1, std_time_T1),'Units','normalized')
  end
  axis auto;
  hold off;
end

% --------------------------------------------------------------------
function s = copy_fields(s, s1)

fname = fieldnames(s1);
for ii=1:length(fname), s.(fname{ii}) = s1.(fname{ii}); end
