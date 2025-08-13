function varargout = ProcessAPFGUI(varargin)
% PROCESSAPFGUI M-file for ProcessAPFGUI.fig
%      PROCESSAPFGUI, by itself, creates a new PROCESSAPFGUI or raises the
%      existing
%      singleton*.
%
%      H = PROCESSAPFGUI returns the handle to a new PROCESSAPFGUI or the handle to
%      the existing singleton*.
%
%      PROCESSAPFGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PROCESSAPFGUI.M with the given input arguments.
%
%      PROCESSAPFGUI('Property','Value',...) creates a new PROCESSAPFGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ProcessAPFGUI_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ProcessAPFGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ProcessAPFGUI

% Last Modified by GUIDE v2.5 17-Feb-2022 13:28:45

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @ProcessAPFGUI_OpeningFcn, ...
    'gui_OutputFcn',  @ProcessAPFGUI_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --------------------------------------------------------------------
function ProcessAPFGUI_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;
handles.ini = inimanage(epr_GetIniPath('ProcessAPFGUI'));
handles.show = 2;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ProcessAPFGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --------------------------------------------------------------------
function figure1_DeleteFcn(hObject, eventdata, handles)

if isfield(handles, 'ini')
    inimanage(epr_GetIniPath('ProcessAPFGUI'), handles.ini);
end

% --------------------------------------------------------------------
function varargout = ProcessAPFGUI_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pbSelectFile_Callback(hObject, eventdata, handles)

dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'SourcePath', 'C:/');

[FileName,PathName] = uigetfile({'*.d01; *.tdms', 'All Formats (*.d01; *.tdms)'; '*.d01', 'SpecMan files (*.d01)'; '*.tdms', 'SpecMan files (*.tdms)'; '*.mat', 'Matlab files (*.mat)'; '*.*', 'All files'},'Load file', old_path);
if PathName ~= 0
    handles.ini.Directories.SourcePath = PathName;
    set(handles.editSourceFile, 'String', fullfile(PathName, FileName));
    guidata(handles.figure1, handles);
end

% --------------------------------------------------------------------
function pbLoadFile_Callback(hObject, eventdata, handles)

src_fname =  get(handles.editSourceFile, 'String');
[~, fname, fext] = fileparts(src_fname);

try
    bkg = str2double(get(handles.eBaseline, 'String'));
catch err
    bkg = -1;
end

set(handles.figure1,'Pointer','watch');drawnow
try
    switch fext
        case '.d01'
            [handles.src.ax, handles.src.raw, handles.src.dsc] = kv_d01read(src_fname);
        case '.tdms'
            [handles.src.ax, handles.src.raw, handles.src.dsc] = kv_smtdmsread(src_fname);
    end
    handles.shots = handles.src.ax.shots;
    handles.src.y =  handles.src.raw;
    
    dim = size(handles.src.y, 2);
    
    dx = diff(handles.src.ax.x);
    handles.src.dx = sum(dx)/length(dx);

    % find noise of the source
    BL = str2double(handles.eBline.String);
    handles.src.noise_idx = size(handles.src.y,1)+(-BL:0);
    Noise = handles.src.raw(handles.src.noise_idx,:);
    Noise = Noise - mean(Noise);
    
    % low pass filter  
    FT = fft(Noise);
    dx = diff(handles.src.ax.x); dx = sum(dx)/length(dx);
    BW = 1/dx/size(Noise,1);
    Pass = floor(10e6/BW);
    FT(Pass+1:end-Pass-1, :) = 0;
    FilteredNoise = ifft(FT);
    
    handles.src.NoiseRMS = std(FilteredNoise);
    
    %correct background
    if bkg > 0
        bl = handles.src.y(:,bkg);
        for ii=1:dim
            handles.src.y(:,ii) = handles.src.y(:,ii) - bl;
        end
        in_opt.baseline_algorithm = 'zero_single';
        in_opt.baseline_area = handles.src.noise_idx;
        handles.src.y = squeeze(kv_baseline_correction(handles.src.y, in_opt));
    end
    
    handles.src.maxy = max(abs(handles.src.y));
    
    handles.Exclude   = zeros(dim, 1);
    if bkg >0, handles.Exclude(bkg) = 1; end
    set(handles.slDataset, 'Min', 1, 'Max', dim, 'Value', 1, 'SliderStep', [1 1]/(dim-1));
    pars = GetFFTPars(handles);
    pars.pft.lshift = -1;
    [handles.amplitude, handles.phase, handles.frequency, outpars] = CalculateAPF(handles.src, pars);
    set(handles.eLShift, 'String', num2str(outpars.pft.lshift))
    max_amp = max(handles.amplitude(~handles.Exclude));
    handles.Exclude = handles.Exclude | handles.amplitude < max_amp*0.05;
    
    guidata(handles.figure1, handles);
    Plot(handles);
    
    disp(' '); disp(['Data file ''',fname,'''is loaded.']);
    set(handles.editSourceFile, 'BackgroundColor', [.95, .95, 1]);
    slDataset_Callback(hObject, eventdata, handles);
    
%     MatrixGUI(FilteredNoise./max(handles.src.maxy))    
catch
    disp(' ');
    disp(['ERROR! Data file ''',src_fname,'''is not loaded.']);
    disp(lasterr); disp(' ');
    set(handles.editSourceFile, 'BackgroundColor', [1, .92, .92]);
end
set(handles.figure1,'Pointer','arrow');drawnow

% --------------------------------------------------------------------
function slDataset_Callback(hObject, eventdata, handles)
Plot(handles);

% --------------------------------------------------------------------
function pars = GetFFTPars(handles)

pars.method = get(handles.pmCalcMethod, 'Value');

pars.pft.data='0_';
pars.pft.awin = 'ham';
pars.pft.awidth = 1;
pars.pft.aalpha = 0.6;
pars.pft.ashift = 0;
pars.pft.zerofill = 2;
pars.pft.rshift = 0;

try
    pars.pft.lshift = str2double(get(handles.eLShift, 'String'));
catch err
    pars.pft.lshift = 124;
end

pars.pft.opt = 'imag';
pars.pft.xshift = 0;
pars.pft.xscale = 1;
pars.pft.cta = 100;

pars.pft.phase0 = 0;
pars.pft.phase1 = 0;
pars.pft.bsline='none';
pars.pft.center = 512;
pars.pft.FOV = 100;

% if isfield(handles, 'frequency')
%   if isfield(handles,'phase')
%     pars.pft.phase0 = handles.phase(dset)+ handles.frequency(dset)*1E6*td*1E-9*360;
%   else
%     pars.pft.phase0 = handles.frequency*1E6*td*1E-9*360;
%   end
% end


% --------------------------------------------------------------------
function Plot(handles)

dset =  fix(get(handles.slDataset, 'Value'));
set(handles.eDataset, 'String', num2str(dset));
if dset <= size(handles.src.y, 2)
    set(handles.eShowAmp, 'String', num2str(handles.amplitude(dset)));
    set(handles.eShowFreq, 'String', sprintf('%4.2f',handles.frequency(dset)));
    set(handles.eShowPhase, 'String', sprintf('%4.2f',handles.phase(dset)));
    set(handles.ePhase, 'String', sprintf('%4.2f',handles.phase(dset)));
    set(handles.cbExclude, 'Value', handles.Exclude(dset));
end

y  = handles.src.y(:, dset);
src_ax = handles.src.ax;
lshift = str2double(get(handles.eLShift, 'String'));

cla(handles.ax);

switch handles.show
    case 1
        show_pts = floor(5e-6 / handles.src.dx);
        idx = max(lshift-show_pts, 1):min(show_pts, length(y));
        plot(src_ax.x(idx), real(y(idx)), src_ax.x(idx), imag(y(idx)), 'Parent', handles.ax); hold on
        axis(handles.ax, 'tight');
        try
            ax = axis(handles.ax);
            plot(src_ax.x(lshift)*[1,1],  ax(4) + [-(ax(4)-ax(3))*.2, 0], 'r-', 'Parent', handles.ax);
        catch err
          disp(err)
        end
        title('Time trace [us]', 'Parent', handles.ax)
    case 2
        src_ax.y = 1;
        pars = GetFFTPars(handles);
        [ax, y] = kv_fft(src_ax, y, pars.pft);
        if(~isempty(strfind(ax.xlabel, 'GHz'))), ax.x = ax.x * 1E3; end
        
        try
            FOV = str2double(get(handles.eFOV, 'String'));
        catch err
            FOV = 25;
        end
        y = y * exp(-1i*handles.phase(dset)*pi/180);
        ROI = ax.x >= handles.frequency(dset) - FOV/2 &  ax.x <= handles.frequency(dset) + FOV/2;
        
        plot(ax.x(ROI), real(y(ROI)), ax.x(ROI), imag(y(ROI)),ax.x(ROI), abs(y(ROI)), 'Parent', handles.ax);
        axis(handles.ax, 'tight');
        title('FFT of time trace [MHz]', 'Parent', handles.ax)
        
        [fw, pars] = calculateFW(ax.x(ROI), real(y(ROI)), 1/2);
        text(pars.xmax, pars.ymax/2*1.1, sprintf('%5.4f G',fw/2.802),'HorizontalAlignment','center')
        plot([pars.x1, pars.x2], pars.ymax/2*[1,1],'r')
        
    case 3
        amp = handles.amplitude(~handles.Exclude);
        frq = handles.frequency(~handles.Exclude);
        plot(frq,amp,'.-', 'Parent', handles.ax); hold on
        if ~handles.Exclude(dset)
            plot(handles.frequency(dset), handles.amplitude(dset), 'r*')
        end
        axis(handles.ax, 'tight');
        
        [frq,sidx] = sort(frq); amp = amp(sidx);
        [mm, m_idx] = max(amp);
        ax = axis(handles.ax);
        plot(frq(m_idx)*[1,1],  ax(4) + [-(ax(4)-ax(3))*.1, 0], 'r-')
        text(frq(m_idx), ax(4)-(ax(4)-ax(3))*.15, sprintf('%f',mm),'HorizontalAlignment','center')
        
        % 3dB point
        [fw, pars] = calculateFW(frq, amp, 1/sqrt(2));
        text(pars.xmax, pars.ymax/sqrt(2)*(0.95), sprintf('%4.2f',fw),'HorizontalAlignment','center')
        plot([pars.x1, pars.x2], pars.ymax/sqrt(2)*[1,1],'r', 'Parent', handles.ax)
        % 6dB point
        [fw, pars] = calculateFW(frq, amp, 1/2);
        text(pars.xmax, pars.ymax/2*(0.95), sprintf('%4.2f',fw),'HorizontalAlignment','center')
        plot([pars.x1, pars.x2], pars.ymax/2*[1,1],'r', 'Parent', handles.ax)
        
        % lorentzian
%         fit = @(x) sqrt(sum((yy - x(1) * real(epr_Lorentzian(frq, x(2), x(3)))).^2));
%         x =  [max(amp)*max(frq), 0, max(frq)/2];
%         xx = fminsearch(fit, x);
%         xx = fminsearch(fit, xx);
%         plot(frq, xx(1)*real(epr_Lorentzian(frq, xx(2), xx(3))), ':')
        
        % sinusoid
        f   = @(x) x(1) * cos(pi*(frq-x(3))*x(2)).^3;
        fit = @(x) sqrt(sum((amp - f(x)).^2));
        x =  [max(amp), 0.05, 0];
        xx = fminsearch(fit, x);
        xx = fminsearch(fit, xx);
        plot(frq, f(xx), 'k:', 'Parent', handles.ax)
        text(pars.xmax, pars.ymax*(0.15), sprintf(':k cos3 %4.2f',0.2081*2/xx(2)),'HorizontalAlignment','center')

        % gauss
        f   = @(x) x(1) * gauss((frq-x(3))*x(2));
        fit = @(x) sqrt(sum((amp - f(x)).^2));
        x =  [max(amp), 0.2, 0];
        xx = fminsearch(fit, x);
        xx = fminsearch(fit, xx);
        plot(frq, f(xx), 'r:')
        text(pars.xmax, pars.ymax*(0.25), sprintf(':r gauss %4.2f',1.6652/xx(2)),'HorizontalAlignment','center')
        
        title('Imager transfer function [MHz]', 'Parent', handles.ax)
    case 4
        plot(handles.frequency(~handles.Exclude), handles.phase(~handles.Exclude),'.-', 'Parent', handles.ax); hold on
        if ~handles.Exclude(dset)
            plot(handles.frequency(dset), handles.phase(dset), 'r*', 'Parent', handles.ax)
        end
        axis(handles.ax, 'tight');
        title('Phase profile [degree]', 'Parent', handles.ax)
    case 5
        src_ax.y = 1;
        pars = GetFFTPars(handles);
        [ax, y] = kv_fft(src_ax, y, pars.pft);
        if(~isempty(strfind(ax.xlabel, 'GHz'))), ax.x = ax.x * 1E3; end
        ax.x = ax.x / 2.802;
        %     y = y * exp(-1i*handles.phase(dset)*pi/180);
        y = abs(y);
        
        try
            FOV = str2double(get(handles.eFOV, 'String'));
        catch err
            FOV = 25;
        end
        FOV = FOV / 2.802;
        ROI = ax.x >= handles.frequency(dset)/2.802 - FOV/2 &  ax.x <= handles.frequency(dset)/2.802 + FOV/2;
        
        plot(ax.x(ROI), real(y(ROI)), ax.x(ROI), imag(y(ROI)), 'Parent', handles.ax);
        HF = [-1.2, -0.625, 0.625, 1.2];
        HFH = max(real(y)) / 15;
        for ii=1:length(HF)
          plot(HF(ii)*[1,1], [0,HFH], 'Parent', handles.ax)
        end
        axis(handles.ax, 'tight');
        title('FFT of time trace [MHz]', 'Parent', handles.ax)
        
        [fw, pars] = calculateFW(ax.x(ROI), real(y(ROI)), 1/2);
        text(pars.xmax, pars.ymax/2*1.1, sprintf('%5.4f G',fw),'HorizontalAlignment','center')
        plot([pars.x1, pars.x2], pars.ymax/2*[1,1],'r', 'Parent', handles.ax)
    case 6
        plot(handles.frequency(~handles.Exclude), handles.src.NoiseRMS(~handles.Exclude),'.-', 'Parent', handles.ax); hold on
        text(.05, .9, sprintf('mean=%6.5f',mean(handles.src.NoiseRMS)), 'Units', 'normalized','HorizontalAlignment','left')
        title('Noise RMS', 'Parent', handles.ax)
        
        %     figure;
        %     subplot(2,1,1); hold on
        %     x = bw / 2.8025;
        %     y = d3(:,dset);
        %     plot(x, y); axis('tight');
        %     [fw, pars] = calculateFW(x, y, 1/2);
        %     text(pars.xmax, pars.ymax/2*1.1, sprintf('%5.4f',fw),'HorizontalAlignment','center')
        %     plot([pars.x1, pars.x2], pars.ymax/2*[1,1],'r')
        %     xlabel('Field [G]');
        %     subplot(2,1,2)
        %     plot(bw(2:end)/ 2.8025, diff(mean(d3, 2))); axis('tight');
        %     title('EPR profile [G]', 'Parent')
end
y  = real(handles.src.y(:, dset));
if lshift < 0, lshift = 10; end
idx = lshift-10:lshift+10;
idx = idx(idx >= 1);
snr = max(y(idx) / mean(handles.src.NoiseRMS));
snrmax = max(max(real(handles.src.y(idx,:)))) / mean(handles.src.NoiseRMS);
text(.75, .9, sprintf('SNR_{%i}=%4.1f',handles.shots, snr), 'Units', 'normalized')
text(.75, .8, sprintf('SNR_{10000}=%4.1f',snr*sqrt(10000/handles.shots)), 'Units', 'normalized')
text(.75, .7, sprintf('SNR/Shot = %4.2f',snrmax*sqrt(1/handles.shots)), 'Units', 'normalized')

% --------------------------------------------------------------------
function ePhase_Callback(hObject, ~, handles)

dset =  fix(get(handles.slDataset, 'Value'));
handles.phase(dset) = str2num(get(handles.ePhase, 'String'));
guidata(hObject, handles);
Plot(handles);

% --------------------------------------------------------------------
function slPhase_Callback(hObject, eventdata, handles)

shift =  fix(get(handles.slPhase, 'Value'));
set(handles.slPhase, 'Value', 0);

dset =  fix(get(handles.slDataset, 'Value'));
handles.phase(dset) = handles.phase(dset) + shift;
guidata(hObject, handles);
Plot(handles);

% --------------------------------------------------------------------
function cbExclude_Callback(hObject, ~, handles)
dset =  fix(get(handles.slDataset, 'Value'));
handles.Exclude(dset) = get(handles.cbExclude, 'Value');
guidata(hObject, handles);
Plot(handles);

% --------------------------------------------------------------------
function pbSave_Callback(~, ~, handles)

output_profile = false;

if output_profile
    disp('APF = [...');
    
    for ii=1:size(handles.phase)
        if ~handles.Exclude(ii)
            fprintf('%7.5f, %4.1f, %6.3f; ...\n', handles.amplitude(ii), ...
                handles.phase(ii), handles.frequency(ii));
        end
    end
    disp('];');
end

dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'SourcePath', 'C:/');

fname = 'cavity_profile';
[fname, directory_name] = uiputfile(...
    {'*.mat', 'MATLAB file (*.mat)'; '*.*',  'All Files (*.*)'}, ...
    'Save template', fullfile(old_path, fname));
if ~isequal(fname,0) && ~isequal(directory_name,0)
    
    file_type = 'CavityProfile';
    idx = ~handles.Exclude;
    Amplitude = handles.amplitude(idx);
    Phase     = handles.phase(idx);
    Frequency = handles.frequency(idx);
    [Frequency, idx] = sort(Frequency);
    Amplitude = Amplitude(idx);
    Phase     = Phase(idx);
    
    save(fullfile(directory_name,fname), 'file_type', 'Amplitude', 'Phase', 'Frequency');
    fprintf('APF profile is saved.\n');
end


% --------------------------------------------------------------------
function pbPrint_Callback(~, ~, handles)

APF = [handles.amplitude'; handles.phase'; handles.frequency']';
idx = ~handles.Exclude;
APF = APF(idx, :);

PrintAPFFIG(1, APF, struct('normalize_amplitude', 1))

% --------------------------------------------------------------------
function Domain_Callback(hObject, ~, handles)
hh = [handles.rbTimeDomain, handles.rbFrequencyDomain, handles.rbProfile, handles.rbPhase, handles.rbEPRshape,handles.rbNoise];
handles.show = find(hh==hObject);

set(hh(hh~=hObject), 'Value', 0);
guidata(hObject, handles);
Plot(handles);

% --------------------------------------------------------------------
function  [amplitude, phase, frequency, outpars] = CalculateAPF(src, pars)

dim = size(src.y, 2);

% estimate zero time
if pars.pft.lshift < 0
    tic
    % get zero time from the first trace
    first_trace = abs(src.y(:,1));
    idx = find(first_trace > max(first_trace)*0.6);
    pars.pft.lshift = idx(end);
    for ii=length(idx)-1:-1:1
        if first_trace(pars.pft.lshift) < first_trace(idx(ii))
            pars.pft.lshift = idx(ii);
        elseif first_trace(pars.pft.lshift)*0.9 > first_trace(idx(ii))
            break;
        end
    end
    
    %get approximate parameters
    max_idx = true(dim-1,1);
    for ii=1:3
        max_idx(1:ii*3) = false;
        max_idx(end-ii*3:end) = false;
        [ax, y] = kv_fft(src.ax, src.y, pars.pft);
        if(~isempty(strfind(ax.xlabel, 'GHz'))), ax.x = ax.x * 1E3; end
        
        in_opt.phase_algorithm = 'max_real_single';
        [out_y, out_pars] = kv_phase_optimize(y, in_opt);
        phase     = out_pars.phase_zero_order*180/pi;
        frequency = ax.x(out_pars.max_idx_all);
        
        dphase = diff(phase);
        dfrequency = diff(frequency);
        idx_minus = dphase < 0 & max_idx;
        idx_plus = dphase > 0 & max_idx;
        d_step = mean(diff(src.ax.x));
        if sum(idx_minus) > sum(idx_plus)
            dpf = mean(dphase(idx_minus)./dfrequency(idx_minus))/1e6/360/d_step;
        else
            dpf = mean(dphase(idx_plus)./dfrequency(idx_plus))/1e6/360/d_step;
        end
        if abs(dpf) > 1000, break; end
        pars.pft.lshift = fix(pars.pft.lshift + dpf);
    end
    toc
end
outpars = pars;

[ax, y] = kv_fft(src.ax, src.y, pars.pft);
if(~isempty(strfind(ax.xlabel, 'GHz'))), ax.x = ax.x * 1E3; end

switch pars.method
    case 1
        in_opt.phase_algorithm = 'max_real_single';
        [out_y, out_pars] = kv_phase_optimize(y, in_opt);
        phase     = out_pars.phase_zero_order*180/pi;
        frequency = ax.x(out_pars.max_idx_all);
        amplitude = max(real(out_y))';
    case 2
        in_opt.phase_algorithm = 'manual';
        in_opt.phase_zero_order = 0;
        [out_y, out_pars] = kv_phase_optimize(y, in_opt);
        phase     = out_pars.phase_zero_order*180/pi;
        frequency = ax.x(out_pars.max_idx_all);
        amplitude = max(real(out_y))';
    case 3
        handles.ShowFit = get(handles.cb_ShowFit,'Value');
        in_opt.phase_algorithm = 'max_real_single';
        [out_y, out_pars] = kv_phase_optimize(y, in_opt);
        phase     = out_pars.phase_zero_order*180/pi;
        % vv are the variables for the lshape function from easyspin
        % package. VV: 1:center, 2:fwhm, 3: gauss/lorentz weight,
        % 4:Real/imaginary phase 5:Overall normalization factor
        trtyl=get(handles.pmSpinProbe, 'Value');
        switch trtyl
            case 1, llw=0.54;
            case 2, llw=0.50;
            case 3, llw=0.45;
            case 4, llw=0.2;
            case 5, llw=0.2;
        end
        
        myfunc = @(vv)lshape(ax.x, vv(1), abs(vv(2)) , 0, 0.5)*vv(3);
        % vv(3): vv(3)^2/(1+vv(3)^2)
        for ii=1:dim
            %             uu = fminsearch( @(uu) sum(-1*real(exp(-i*uu)*out_y(:,ii)),1)
            %             );
            options.MaxFunEvals = 10000;
            % Median Filtering
            outer_y(:,ii) = out_y(:,ii) - medfilt1(out_y(:,ii),300); % 200 for now... change later
            %             outer_y(:,ii) = out_y(:,ii) - mean(out_y,2);
            %             [garb,b]=max(outer_y(:,ii));
            uu0 = [sum(abs(outer_y(:,ii))/100,1),0]; % amp, phase
            [garb, c]=max(abs(diff(convn(myfunc([0,llw,uu0(1)]),real(outer_y(:,ii)),'same'))));
            uu0 = [ax.x(c), llw, sum(abs(outer_y(:,ii))/100,1),0];
            uu=  fminsearch(@(uu) sum((real(exp(1i*uu(4))*outer_y(:,ii))-myfunc([uu(1), uu(2), uu(3)])).^2,1),uu0);
            uu40=uu(4); %phase
            uu(4) = fminsearch(@(ww) sum((real(exp(1i*ww)*out_y(:,ii))-myfunc([uu(1), uu(2), uu(3)])).^2,1),uu40);
            %             disp(sprintf('Peak:%e Amp:%e    Phase:%e', uu(1), uu(2), phase(ii,1)+uu(3)*180/pi));
            % atan2(sum(imag(y(:,ii)),1),sum(real(y(:,ii)),1)) for phase
            phase(ii,1)     = phase(ii) + (uu(4))*180/pi;
            frequency(ii,1) = uu(1);
            amplitude(ii,1) = uu(3);
            if handles.ShowFit
                figure(101), plot(ax.x,real(exp(1i*uu(4))*out_y(:,ii)), ax.x, real(myfunc([uu(1), uu(2), uu(3)])));
                xlim([ax.x(c)-3 ax.x(c)+3]); % in MHz
            end
        end
        if handles.ShowFit, close(101); end
end

% --------------------------------------------------------------------
function cb_ShowFit_Callback(~, ~, handles)
Plot(handles);

% --------------------------------------------------------------------
function Calculate(handles)

[handles.amplitude, handles.phase, handles.frequency] = CalculateAPF(handles.src, GetFFTPars(handles));

guidata(handles.figure1, handles);
Plot(handles);
% --------------------------------------------------------------------

function [fw, pars] = calculateFW(x,y,threshold)
% low resolution
[x,idx] = unique(x); y = y(idx);

mm = max(y);
off6dBmin = find(y > mm * threshold / 2, 1, 'first');
off6dBmax = find(y > mm * threshold / 2, 1, 'last');

x1 = linspace(x(off6dBmin), x(off6dBmax), 1000)';
y1 = interp1(x,y,x1, 'spline');

[pars.ymax, m_idx] = max(y1);
pars.xmax = x1(m_idx);
min_idx = find(y1 > pars.ymax * threshold, 1, 'first');
max_idx = find(y1 > pars.ymax * threshold, 1, 'last');
pars.x1 = x1(min_idx);
pars.x2 = x1(max_idx);
fw = abs(pars.x2 - pars.x1);

% -------------------------------------------------------------------------
function res = sinc(x)
idx = x == 0;
res = ones(size(x));
res(~idx) = sin(x) ./ x;

% -------------------------------------------------------------------------
function res = gauss(x)
res = exp(-x.^2);
