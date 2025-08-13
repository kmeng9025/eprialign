function varargout = TimelineGUI(varargin)
% TIMELINEGUI MATLAB code for TimelineGUI.fig
%      TIMELINEGUI, by itself, creates a new TIMELINEGUI or raises the existing
%      singleton*.tim
%
%      H = TIMELINEGUI returns the handle to a new TIMELINEGUI or the handle to
%      the existing singleton*.proc
%
%      TIMELINEGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TIMELINEGUI.M with the given input arguments.
%
%      TIMELINEGUI('Property','Value',...) creates a new TIMELINEGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before TimelineGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to TimelineGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help TimelineGUI

% Last Modified by GUIDE v2.5 23-Jun-2023 13:44:27

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @TimelineGUI_OpeningFcn, ...
  'gui_OutputFcn',  @TimelineGUI_OutputFcn, ...
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
% --- Executes just before TimelineGUI is made visible.
function TimelineGUI_OpeningFcn(hObject, ~, handles, varargin)

% Choose default command line output for TimelineGUI
handles.output = hObject;

% Update table of samples
calibrations = epr_spin_probe_calibration;
set(handles.pmProbe, 'string', calibrations);

handles.FileList = {};

handles.FitModes{1}.Template = 'NCI_TEST'; %FID
handles.FitModes{1}.Method = 'FID';
handles.FitModes{1}.Parameter = 'T2*';
handles.FitModes{2}.Template = 'T2'; %ESE
handles.FitModes{2}.Method = 'ESE';
handles.FitModes{2}.Parameter = 'T2';
handles.FitModes{3}.Template = 'T1'; %IRESE
handles.FitModes{3}.Method = 'IRESE';
handles.FitModes{3}.Parameter = 'T1';
FitMode(handles, 3);
handles = guidata(hObject);

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes TimelineGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --------------------------------------------------------------------
% --- Outputs from this function are returned to the command line.
function varargout = TimelineGUI_OutputFcn(~, ~, handles)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pbReload_Callback(hObject, eventdata, handles)
FindFiles(handles);
handles = guidata(handles.figure1);
pbRun_Callback(hObject, eventdata, handles);

% --------------------------------------------------------------------
function pbRun_Callback(hObject, ~, handles)
ProcessData(handles, true);

% --------------------------------------------------------------------

function ProcessData(handles, isreprocess)
h = waitbar(0,'Please wait...');
id = tic;
for ii=1:length(handles.FileList)
  if isreprocess || ~handles.FileList{ii}.isfitted
    handles.FileList{ii}.tvec = datevec(now);
    handles.FileList{ii}.amp = 0;
    handles.FileList{ii}.t1  = 1e-6;
    handles.FileList{ii}.loaded = false;
    try
      % load
      filename = fullfile(handles.FileList{ii}.directory, handles.FileList{ii}.name);
      [~,~,ext] = fileparts(filename);
      switch ext
        case '.exp'
          [ax,y, dsc] = kv_d01read(filename);
          x = ax.x;
          y = real(y)';
          handles.FileList{ii}.datetime = ax.FinishTime;
          handles.FileList{ii}.loaded = true;
        case '.tdms'
          [ax,y, dsc] = epr_loadsmtdms(filename);
          ttt = dir(filename);
          handles.FileList{ii}.datetime = datenum(datetime(ttt.date));
          x = ax.x;
          handles.FileList{ii}.loaded = true;
        otherwise
      end
      if ~handles.FileList{ii}.loaded, continue; end
      
      aliases = safeget(dsc, 'devices_aliases', []);
      
      monitors = safeget(dsc, 'devices_monitors', []);
      mix = safeget(dsc, 'devices_MIX', []);
      handles.FileList{ii}.temperature1 = str2val(safeget(monitors, 'TEMPERATURE1', '0 C'));
      handles.FileList{ii}.temperature2 = str2val(safeget(monitors, 'TEMPERATURE2', '0 C'));
      handles.FileList{ii}.tempcont = str2val(safeget(monitors, 'TEMPCONT', '0 C'));
      handles.FileList{ii}.flow1 = str2val(safeget(monitors, 'FLOW1', '0 sccm'));
      handles.FileList{ii}.flow2 = str2val(safeget(monitors, 'FLOW2', '0 sccm'));
      handles.FileList{ii}.flow3 = str2val(safeget(monitors, 'FLOW3', '0 sccm'));
      handles.FileList{ii}.gain = str2val(safeget(mix, 'Gain', '25 dB'));
      
      % fit
      switch handles.FileList{ii}.fitmode
        case 3 %IRESE
          handles.FileList{ii}.x = x;
          handles.FileList{ii}.phase = 0;
          if handles.cbPhaseData.Value
            % Baseline
            sz = length(y);
            % Phase
            s_idx = floor(sz*0.8):sz;
            handles.FileList{ii}.phase = atan2(mean(imag(y(s_idx))), mean(real(y(s_idx))));
            handles.FileList{ii}.y = phase_zero_order(y(:), handles.FileList{ii}.phase);
          else
            handles.FileList{ii}.y = y(:);
          end
          if handles.cbRemoveOutlier.Value == 1
            [handles.FileList{ii}.amp, handles.FileList{ii}.t1, handles.FileList{ii}.inv] = fit_recovery_3par_clean(real(handles.FileList{ii}.y'), handles.FileList{ii}.x');
          else
            [handles.FileList{ii}.amp, handles.FileList{ii}.t1, handles.FileList{ii}.inv] = fit_recovery_3par(real(handles.FileList{ii}.y'), handles.FileList{ii}.x');
          end
          handles.FileList{ii}.isfitted = true;
        case 2 %ECHO
          handles.FileList{ii}.x = x*2;
          handles.FileList{ii}.inv = 0;
          handles.FileList{ii}.phase = 0;
          if handles.cbPhaseData.Value
            handles.FileList{ii}.phase = atan2(imag(y(1)), real(y(1)));
          end
          handles.FileList{ii}.y = phase_zero_order(y(:), handles.FileList{ii}.phase);
          [handles.FileList{ii}.amp, handles.FileList{ii}.t1] = fit_exp_no_offset(real(handles.FileList{ii}.y'), handles.FileList{ii}.x');
          handles.FileList{ii}.isfitted = true;
        case 1 %FID
          Toff = 0.316e-6;
          block_idx = 35 + 5;
          handles.FileList{ii}.x = x + Toff;
          handles.FileList{ii}.inv = 0;
          handles.FileList{ii}.phase = 0;
          handles.FileList{ii}.block_idx = 35;
          if handles.cbPhaseData.Value
            handles.FileList{ii}.phase = atan2(imag(y(block_idx)), real(y(block_idx)));
          end
          handles.FileList{ii}.y = phase_zero_order(y(:), handles.FileList{ii}.phase);
          x = handles.FileList{ii}.x(block_idx:end);
          y = real(handles.FileList{ii}.y(block_idx:end));
          [handles.FileList{ii}.amp, handles.FileList{ii}.t1, handles.FileList{ii}.off] = fit_exp_offset(y', x');
          
          n = 32384;
          yyy = y; yyy(length(y):n) = y(end); % zero filling
          yyy = yyy - y(end);
          FT = real(fftshift(fft(yyy)));
          dt = mean(diff(x));
          dFT = 1/dt/length(yyy);
          FTnorm = FT / max(FT);
          above = find(FTnorm >= 0.5);
          %         handles.FileList{ii}.FWHH = numel(find(FT > max(FT)/2))*dFT / 2.802E6;
          x1 = above(1)-1; x2 = above(1); y1 = FTnorm(x1); y2 = FTnorm(x2);
          left = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1);
          x1 = above(end); x2 = above(end)+1; y1 = FTnorm(x1); y2 = FTnorm(x2);
          right = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1);
          handles.FileList{ii}.FWHH = (right - left)*dFT / 2.802E6;
          handles.FileList{ii}.isfitted = true;
      end
    catch err
      handles.FileList{ii}.x = 0;
      handles.FileList{ii}.y = 0;
      handles.FileList{ii}.inv = 0;
      handles.FileList{ii}.phase =0;
      handles.FileList{ii}.amp = 0;
      handles.FileList{ii}.t1 =0;
      handles.FileList{ii}.flow1 = 0;
      handles.FileList{ii}.flow2 = 0;
      handles.FileList{ii}.flow3 = 0;
      handles.FileList{ii}.isfitted = false;
      handles.FileList{ii}.datetime = now;
      disp('error');
    end
    waitbar(ii/length(handles.FileList),h)
  end
end

guidata(handles.figure1, handles);
delete(h);
toc(id);

% visulaize
Show(handles);

% --------------------------------------------------------------------
function [val, unit, pref, pref_val] = str2val(str)
prefix = ['p','n', 'u', 'm', 'k', 'M', 'G', 'T'];
koeff  = [1E-12, 1E-9, 1E-6, 1E-3, 1E3, 1E6, 1E9, 1E12];
pref = ''; pref_val = 1;

res = regexp(str, '(?<number>[0-9.eE-+]+)\s*(?<unit>\w+)*', 'names');

if ~isempty(res)
  res = res(1);
  val = str2double(res.number);
  if isfield(res, 'number'), unit = res.unit; else, unit = ''; end
  if length(unit) > 1
    if ~isempty(unit)
      kk = strfind(prefix, unit(1));
      if ~isempty(kk)
        val = val * koeff(kk);
        unit = unit(2:end);
        pref = prefix(kk);
        pref_val = koeff(kk);
      end
    end
  end
end

% -------------------------------------------------------------------------
function out_y = phase_zero_order(in_y, in_phase)
out_y = in_y.*exp(-1i*in_phase);

% --------------------------------------------------------------------
function Show(handles)

Show2(handles);

FitMode = cellfun(@(x) x.fitmode, handles.FileList);
method =  handles.FitModes{handles.FitMode}.Method;
parameter =  handles.FitModes{handles.FitMode}.Parameter;

idx = FitMode == handles.FitMode;
FileList = handles.FileList(idx);

if isempty(FileList), return; end
tm = cellfun(@(x) x.datetime, FileList);
xx = (tm - tm(1))*24; %% hours

% selected item
sel = handles.lbFiles.Value;
xxsel = xx(sel);

cla(handles.axes1);
yyunit = '';
switch handles.pmDisplay.Value
  case 6 % trace
    if length(sel) > 1, sel = sel(1); end
    switch method
      case 'IRESE'
        pars = [FileList{sel}.amp, 1./FileList{sel}.t1, FileList{sel}.inv];
        [~,~,~,~,~,ff] = fit_recovery_3par([],[]);
      case 'ESE'
        [~,~,~,~,ff] = fit_exp_no_offset([],[]);
        pars = [FileList{sel}.amp, 1./FileList{sel}.t1];
      case 'FID'
        [~,~,~,~,~,ff] = fit_exp_offset([],[]);
        pars = [FileList{sel}.amp, 1./FileList{sel}.t1,FileList{sel}.off];
    end
    plot(FileList{sel}.x*1e6, real(FileList{sel}.y), FileList{sel}.x*1e6, imag(FileList{sel}.y),...
      FileList{sel}.x*1e6, ff(pars,FileList{sel}.x),...
      'Parent', handles.axes1); hold(handles.axes1, 'off');
    axis(handles.axes1, 'tight')
    legend(handles.axes1, {'real [us]', 'imag  [us]', 'fit  [us]'});
    grid(handles.axes1, 'on');
  case 8 % phase
    val = cellfun(@(x) x.phase, FileList) * 180/pi;
    valstr = 'phase';
    yyunit = 'deg';
  case 9 % tempearture1
    val = cellfun(@(x) x.temperature1, FileList);
    valstr = 'temp';
    yyunit = 'C';
  case 10 % tempearture2
    val = cellfun(@(x) x.temperature2, FileList);
    valstr = 'temp';
    yyunit = 'C';
  case 11 % tempearture cont
    val = cellfun(@(x) x.tempcont, FileList);
    valstr = 'temp';
    yyunit = 'C';
  case 12 % flow1
    val = cellfun(@(x) x.flow1, FileList);
    valstr = 'flow1';
    yyunit = 'sccm';
  case 13 % flow2
    val = cellfun(@(x) x.flow2, FileList);
    valstr = 'flow2';
    yyunit = 'sccm';
  case 14 % flow3
    val = cellfun(@(x) x.flow3, FileList);
    valstr = 'flow3';
    yyunit = 'sccm';
  case {1,2,3,4,5,7}
    probe = handles.pmProbe.String{handles.pmProbe.Value};
    
    yy = [];
    yyunit = '';
    target_precision = 1;
    switch handles.pmDisplay.Value
      case 1 % pO2[torr]
        fprintf('Calibration selected: %s\n', probe);
        clb = epr_spin_probe_calibration(probe);
        disp(clb)
        t1 = cellfun(@(x) x.t1, FileList);
        t1(t1 > 14e-6) = 14e-6;
        t1(t1 < 0.15e-6) = 0.15e-6;
        yysel = t1(sel);
        [xx, sidx] = sort(xx);
        yy = T2O2(handles, t1(sidx), clb);
        yysel = T2O2(handles, yysel, clb);
        yylegend = [method,' pO2 [torr]'];
        yyunit = 'torr';
        target_precision = 1;
      case 2 % pO2[%]
        fprintf('Calibration selected: %s\n', probe);
        clb = epr_spin_probe_calibration(probe);
        disp(clb)
        t1 = cellfun(@(x) x.t1, FileList);
        t1(t1 > 14e-6) = 14e-6;
        t1(t1 < 0.15e-6) = 0.15e-6;
        yysel = t1(sel);
        [xx, sidx] = sort(xx);
        yy = T2O2(handles, t1(sidx), clb);
        yy = yy / 760 * 100;
        yysel = T2O2(handles, yysel, clb);
        yysel = yysel / 760 * 100;
        yylegend = [method,' pO2 [%]'];
        yyunit = '%';
        target_precision = 0.2;
      case 3 % T1[us]
        t1 = cellfun(@(x) x.t1, FileList) * 1e6;
        yysel = t1(sel);
        [xx, sidx] = sort(xx);
        yy = t1(sidx);
        yylegend = [parameter,' [us]'];
        yyunit = 'us';
        target_precision = 0.05;
      case 4 % R [Ms-1]
        t1 = cellfun(@(x) x.t1, FileList) * 1e6;
        yysel = 1./t1(sel);
        [xx, sidx] = sort(xx);
        yy = 1./t1(sidx);
        parameter(parameter == 'T') = 'R';
        yylegend = [parameter, '  [Ms^{-1}]'];
        yyunit = 'Ms^{-1}';
        target_precision = 0.02;
      case 5
        if handles.FitMode == 1
          FWHH = cellfun(@(x) x.FWHH, FileList) * 1000;
          yysel = FWHH(sel);
          [xx, sidx] = sort(xx);
          yy = FWHH(sidx);
          yylegend = 'FWHH [mG]';
        else
          cc = 1e3/(2*pi*2.802E6);
          t1 = cellfun(@(x) x.t1, FileList);
          yysel = 1./t1(sel) * cc;
          [xx, sidx] = sort(xx);
          yy = 1./t1(sidx) * cc;
          yylegend = 'LLW [mG]';
        end
        yyunit = 'mG';
      case 7 % amplitude
        yy = cellfun(@(x) x.amp, FileList);
        yysel = yy(sel);
        [xx, sidx] = sort(xx);
        yy = yy(sidx);
        yylegend = 'Amp [a.u.]';
        yyunit = 'a.u.';
        gain_factor = exp((cellfun(@(x) x.gain, FileList) - 25)/20); % 25dB is default
        gain_factor = gain_factor(sidx);
        yy = yy./gain_factor;
    end
    
    plot(xx, yy, '.-', 'Parent', handles.axes1); hold(handles.axes1, 'on');
    plot(xxsel, yysel, 'o', 'Parent', handles.axes1); hold(handles.axes1, 'off');
    legend(handles.axes1,yylegend,'Location','north');
    
    if numel(yysel) > 1
      text(0.4, 0.1, sprintf('mean_{%i}=%4.2f(%s%4.2f)%s',numel(yysel),mean(yysel),char(177),std(yysel),yyunit), 'units', 'normalized','FontSize', 16, 'Parent', handles.axes1);
    else
      text(0.4, 0.1, sprintf('val=%5.3f%s',mean(yysel),yyunit), 'units', 'normalized','FontSize', 16, 'Parent', handles.axes1);
    end
    
    if  numel(yysel) < 3
    elseif  numel(yysel) < 10
      % linear trend calculator
      % Fit Data
      b = polyfit(xxsel,yysel, 1);
      fr = polyval(b, xxsel);
      hold(handles.axes1, 'on'); plot(xxsel,fr,'-', 'Parent', handles.axes1);
      text(0.4, 0.05, sprintf('linear trend %4.2f ??/hour',b(1)), 'units', 'normalized','FontSize', 16, 'Parent', handles.axes1);
    elseif  numel(yysel) >= 10
      %       trend calculation
      % define fitting function
      fit_recovery = @(x,t) x(1) + x(3)*exp(-(t-x(4))*x(2));
      fit_fun = @(x) sqrt(sum((yysel - fit_recovery(x,xxsel)).^2));
      rangex = max(xxsel)-min(xxsel);
      if yysel(1) > yysel(end)
        xres = fminsearch(fit_fun,[min(yysel),5/rangex,(max(yysel)-min(yysel)),min(xxsel)]);
        hold(handles.axes1, 'on');
        displayx = linspace(min(xxsel)-rangex/2,max(xxsel)+rangex/2, 40);
        res = fit_recovery(xres,displayx);
        res_idx = res < max(yy);
        plot(displayx(res_idx), res(res_idx),'-', 'Parent', handles.axes1);
        plot([min(xxsel)-rangex/2,max(xxsel)+rangex/2],xres(1)*[1,1],':', 'Parent', handles.axes1); hold(handles.axes1, 'off');
        
        fit_fun = @(x) sqrt(sum((xres(1) + target_precision - fit_recovery(xres,x)).^2));
        x1res = fminsearch(fit_fun,min(xxsel));
        text(0.4, 0.05, sprintf('error below %4.2f at %4.2fh (->%4.2f T=%4.2f[1/h])',target_precision,x1res,xres(1),1/xres(2)), 'units', 'normalized','FontSize', 16, 'Parent', handles.axes1);
      else
        xres = fminsearch(fit_fun,[max(yysel),5/rangex,-(max(yysel)-min(yysel)),min(xxsel)]);
        hold(handles.axes1, 'on');
        displayx = linspace(min(xxsel)-rangex/2,max(xxsel)+rangex/2, 40);
        plot(displayx, fit_recovery(xres,displayx),'-', 'Parent', handles.axes1);
        plot([min(xxsel)-rangex/2,max(xxsel)+rangex/2],xres(1)*[1,1],':', 'Parent', handles.axes1); hold(handles.axes1, 'off');
        
        fit_fun = @(x) sqrt(sum((xres(1)- target_precision - fit_recovery(xres,x)).^2));
        x1res = fminsearch(fit_fun,[min(xxsel)]);
        text(0.4, 0.05, sprintf('error below %4.2f at %4.2fh (->%4.2f T=%4.2f[1/h])',target_precision,x1res,xres(1),1/xres(2)), 'units', 'normalized', 'Parent', handles.axes1);
      end
    end
    text(0.01, 0.98, sprintf('%s', datestr(min(tm))), 'units', 'normalized', 'Parent', handles.axes1,'FontSize', 8);
    text(0.99, 0.98, sprintf('%s', datestr(max(tm))), 'units', 'normalized', 'Parent', handles.axes1,'FontSize', 8, 'HorizontalAlignment','right');
    grid(handles.axes1,'on');
    return; % end of relaxaton case clause
  case 15
    t1 = cellfun(@(x) x.t1, FileList) * 1e6;
    yysel = 1./t1(sel);
    [~, sidx] = sort(xx);
    yy = 1./t1(sidx);
    flow2 = cellfun(@(x) x.flow2, FileList); % N2
    flow3 = cellfun(@(x) x.flow3, FileList); % 21%O2 + N2
    Opc3 = 21;
    O2 = flow3 * Opc3./(flow2 + flow3) * 7.60; % torr
   
    idx = cellfun(@(x) ~contains(x.name, 'SETUP'), FileList, 'UniformOutput', true);
    yyfit  = yy(idx);
    O2fit  = O2(idx);
    
    plot(O2fit, yyfit, '.', 'Parent', handles.axes1); hold(handles.axes1, 'on'); 

    if numel(yysel) > 5
      plot(O2(sel), yysel, '.', 'Parent', handles.axes1);
      text(0.4, 0.1, sprintf('mean_{%i}=%4.2f(%s%4.2f)%s',numel(yysel),mean(yysel),char(177),std(yysel),yyunit), 'units', 'normalized','FontSize', 16, 'Parent', handles.axes1);
      idxsel = idx(sel);
      p = polyfit(O2(idxsel),1./t1(idxsel),1);
      f = polyval(p, [min(O2), max(O2)]);
    else
      plot(O2(sel), yysel, 'o', 'Parent', handles.axes1);
      text(0.4, 0.1, sprintf('val=%5.3f%s',mean(yysel),yyunit), 'units', 'normalized','FontSize', 16, 'Parent', handles.axes1);
      p = polyfit(O2fit,yyfit,1);
      f = polyval(p, [min(O2), max(O2)]);
    end
    
    plot([min(O2), max(O2)], f, ':', 'Parent', handles.axes1);
    text(0.4, 0.05, sprintf('fit: %6.4f[Ms^{-1}] %6.4f[torr/Ms^{-1}]',p(2), 1/p(1)), 'units', 'normalized','FontSize', 16, 'Parent', handles.axes1);
    
    hold(handles.axes1, 'off');
    return;
end

if exist('val', 'var')
  yysel = val(sel);
  plot(xx, val, 'Parent', handles.axes1); hold(handles.axes1,'on') ; 
  plot(xxsel, yysel, 'o', 'Parent', handles.axes1); hold(handles.axes1,'off') 
  axis(handles.axes1,'tight');
  
  if numel(yysel) > 1
    text(0.4, 0.1, sprintf('mean_{%i}=%4.2f(%s%4.2f)%s',numel(yysel),mean(yysel),char(177),std(yysel),yyunit), 'units', 'normalized','FontSize', 16, 'Parent', handles.axes1);
  else
    text(0.4, 0.1, sprintf('val=%5.3f%s',mean(yysel),yyunit), 'units', 'normalized','FontSize', 16, 'Parent', handles.axes1);
  end
  text(0.01, 0.98, sprintf('%s', datestr(min(tm))), 'units', 'normalized', 'Parent', handles.axes1,'FontSize', 8);
  text(0.99, 0.98, sprintf('%s', datestr(max(tm))), 'units', 'normalized', 'Parent', handles.axes1,'FontSize', 8, 'HorizontalAlignment','right');
end

% --------------------------------------------------------------------
function inp = uncell(inp)
if iscell(inp), inp = inp{1}; end

% --------------------------------------------------------------------
function Show2(handles)

FitMode = cellfun(@(x) x.fitmode, handles.FileList);
method =  handles.FitModes{handles.FitMode}.Method;

idx = FitMode == handles.FitMode;
FileList = handles.FileList(idx);

if isempty(FileList), return; end
sel = handles.lbFiles.Value;
tm = cellfun(@(x) x.datetime, FileList);
tmsel = tm(sel);
if length(sel) > 1, sel = sel(1); end

switch method
  case 'IRESE'
    pars = [FileList{sel}.amp, 1./FileList{sel}.t1, FileList{sel}.inv];
    [~,~,~,~,~,ff] = fit_recovery_3par([],[]);
  case 'ESE'
    [~,~,~,~,ff] = fit_exp_no_offset([],[]);
    pars = [FileList{sel}.amp, 1./FileList{sel}.t1];
  case 'FID'
    [~,~,~,~,~,ff] = fit_exp_offset([],[]);
    pars = [FileList{sel}.amp, 1./FileList{sel}.t1,FileList{sel}.off];
end
plot(FileList{sel}.x*1e6, real(FileList{sel}.y), FileList{sel}.x*1e6, imag(FileList{sel}.y),...
  FileList{sel}.x*1e6, ff(pars,FileList{sel}.x),...
  'Parent', handles.axes2); hold(handles.axes2, 'off');
text(0.95, 0.2, sprintf('%s', datestr(min(tmsel))), 'units', 'normalized', 'Parent', handles.axes2,'FontSize', 8,'HorizontalAlignment', 'right');
axis(handles.axes2, 'tight')
grid(handles.axes2, 'on')
legend(handles.axes2, {'real', 'imag', 'fit'});

% --------------------------------------------------------------------
function handles = FindFolder(handles, folder)

ls = dir(folder);
for ii=1:length(ls)
  handles.FileList{end+1}.directory = folder;
  handles.FileList{end}.isfitted = false;
  handles.FileList{end}.name = ls(ii).name;
  handles.FileList{end}.fitmode = 0;
  for kk=1:length(handles.FitModes)
    if contains(handles.FileList{end}.name, handles.FitModes{kk}.Template)
      handles.FileList{end}.fitmode = kk;
      break;
    end
  end
end

% --------------------------------------------------------------------
function FindFiles(handles)

handles.FileList = {};
folders = handles.lbFolders.String;
for jj=1:length(folders)
  handles = FindFolder(handles, folders{jj});
end

guidata(handles.figure1, handles);
UpdateListBox(handles);

% --------------------------------------------------------------------
function UpdateListBox(handles)
str = {};
for ii=1:length(handles.FileList)
  if handles.FileList{ii}.fitmode == handles.FitMode
    str{end+1} = sprintf('%s', handles.FileList{ii}.name);
  end
end
set(handles.lbFiles, 'String', str, 'Value', 1);

% --------------------------------------------------------------------
function pbPlot_Callback(~, ~, handles)
Show(handles);

% --------------------------------------------------------------------
function lbFiles_Callback(~, ~, handles)
Show(handles);

% --------------------------------------------------------------------
function pmDisplay_Callback(~, ~, handles)
Show(handles);

% --------------------------------------------------------------------
function pbAddFolder_Callback(hObject, ~, handles)

[PathName] = uigetdir(safeget(handles, 'DefaultPath', ''));
if ~isequal(PathName, 0)
  str = handles.lbFolders.String;
  str{end+1} = PathName;
  set(handles.lbFolders, 'String', str, 'Value', length(str))
  handles.DefaultPath = PathName;
  handles = FindFolder(handles, PathName);
  guidata(handles.figure1, handles);
  ProcessData(handles, false);
  UpdateListBox(handles);
end

% --------------------------------------------------------------------
function pbRemoveFolder_Callback(hObject, ~, handles)
str = handles.lbFolders.String;
idx = handles.lbFolders.Value;
if idx <= length(str)
  folder_to_remove = str{idx};
  str(idx) = [];
  files_to_keep = true(length(handles.FileList), 1);
  for ii=1:length(handles.FileList)
    if contains(handles.FileList{ii}.directory, folder_to_remove)
      files_to_keep(ii) = false;
    end
  end
  handles.FileList = handles.FileList(files_to_keep);
  guidata(handles.figure1, handles);
end
set(handles.lbFolders, 'String', str, 'Value', min(idx, length(str)))
Show(handles);

% --------------------------------------------------------------------
function mFileExportExcel_Callback(~, ~, handles)

FitMode = cellfun(@(x) x.fitmode, handles.FileList);

idx = FitMode == handles.FitMode;
FileList = handles.FileList(idx);

tm = cellfun(@(x) x.datetime, FileList);
dt = (tm - tm(1))*24; %% hours

probe = handles.pmProbe.String{handles.pmProbe.Value};
fprintf('Calibration selected: %s\n', probe);
clb = epr_spin_probe_calibration(probe);

[etime, sidx] = sort(dt);

t1 = cellfun(@(x) x.t1, FileList);
t1 = t1(sidx);
O2 = T2O2(handles, t1, clb);

name = cellfun(@(x) x.name, FileList, 'UniformOutput', false);
name = name(sidx);

res = cell(length(name),3);
for ii=1:length(name)
  res{ii,1} = name{ii};
  res{ii,2} = etime(ii);
  res{ii,3} = O2(ii);
end

[filename, pathname, choice] = uiputfile({'*.xls','Excel file (*.xls)';'*.csv','CSV file (*.csv)'}, 'Pick a file');
if isequal(filename,0) || isequal(pathname,0)
  %   disp('User pressed cancel')
else
  filename = fullfile(pathname, filename);
  if exist(filename, 'file'), delete(filename); end
  switch choice
    case 1, xlswrite(filename,res);
    case 2
      h = fopen(filename, 'w+');
      for ii=1:length(name)
        fprintf(h, '%s, %4.2f, %6.2f\n', res{ii,1}, res{ii,2}, res{ii,3});
      end
      fclose(h);
  end
  fprintf('Results are written to %s.\n', filename);
end

% --------------------------------------------------------------------
function O2=T2O2(handles, T, clb)

method =  handles.FitModes{handles.FitMode}.Method;

po2_struct = [];
f = 1/pi/2/2.802*1000; % conversion of MS-1 to mG
if contains(method, 'IRESE')
  po2_struct.LLW_zero_po2 = clb.interceptT1 * f;
  po2_struct.Torr_per_mGauss =  clb.slopeT1 / f;
else
  po2_struct.LLW_zero_po2 = clb.interceptT2 * f;
  po2_struct.Torr_per_mGauss =  clb.slopeT2 / f;
end
O2 = epr_T2_PO2(T*1e6, 0, [], po2_struct);

% --------------------------------------------------------------------
function mContextDeleteItem_Callback(~, ~, handles)
FitMode = cellfun(@(x) x.fitmode, handles.FileList);
sel = handles.lbFiles.Value;
idx = FitMode == handles.FitMode;
FileList = handles.FileList(idx);

if sel > 0
  File = FileList(sel);
  idx = true(size(handles.FileList));
  for ii=1:length(handles.FileList)
    for jj=1:length(File)
      if contains(handles.FileList{ii}.name, File{jj}.name)
        idx(ii)=false;
        break;
      end
    end
  end
  handles.FileList = handles.FileList(idx);
  guidata(handles.figure1, handles);
  UpdateListBox(handles);
  Show(handles)
end

% --------------------------------------------------------------------
function pmProbe_Callback(~, ~, handles)
Show(handles);

% --------------------------------------------------------------------
function mToolMonitor_Callback(~, ~, handles)
folders = handles.lbFolders.String;
folder = folders{1};

ls = dir(fullfile(folder, '*.tdms'));
expT = zeros(length(ls),1);
for ii=1:length(ls)
  expT(ii)=ls(ii).datenum;
end

%filename = {'TEMPCONT.dat','FLOW1.dat','FLOW2.dat'};
filename = {'TEMPCONT.dat','TEMPERATURE1.dat','PWM.dat'};

figure;
for mon=1:3
  fid = fopen(fullfile(folder, filename{mon}), 'r');
  if fid ~= -1
    i = 1;
    X = []; Y = [];
    while ~feof(fid)
      l = fgetl(fid);
      A = sscanf(l, '%d:%d:%d:%d:%d:%d %f');
      X(i) = datenum(A(1),A(2),A(3),A(4),A(5),A(6));
      Y(i) = A(7);
      i = i+1;
    end
    fclose(fid);
  end
  
  subplot(3,1,mon)
  REF = X(1);
  X = X - REF; ExpTX = expT - REF;
  plot(X*24, Y); hold on
  p = [min(Y), max(Y)];
  for jj=1:length(expT)
    plot(ExpTX(jj)*24*[1, 1], p, 'k');
  end
  xlabel('Hours');
  title(filename{mon});
  axis tight
  grid on
end

% --------------------------------------------------------------------
function pbIRESE_Callback(~, ~, handles)
FitMode(handles, 'IRESE')

% --------------------------------------------------------------------
function pbESE_Callback(~, ~, handles)
FitMode(handles, 'ESE')

% --------------------------------------------------------------------
function pbFID_Callback(~, ~, handles)
FitMode(handles, 'FID')

% --------------------------------------------------------------------
function FitMode(handles, mode)

switch mode
  case 'FID'
    handles.FitMode = 1;
    set(handles.pbFID, 'background', [0,1,0])
    set([handles.pbIRESE, handles.pbESE], 'background', [0.9,0.9,0.9])
  case 'ESE'
    handles.FitMode = 2;
    set(handles.pbESE, 'background', [0,1,0])
    set([handles.pbIRESE, handles.pbFID], 'background', [0.9,0.9,0.9])
  otherwise % IRESE
    handles.FitMode = 3;
    set(handles.pbIRESE, 'background', [0,1,0])
    set([handles.pbESE, handles.pbFID], 'background', [0.9,0.9,0.9])
end

handles.eTemplate.String = handles.FitModes{handles.FitMode}.Template;
guidata(handles.figure1, handles);
UpdateListBox(handles);
Show(handles);

% --------------------------------------------------------------------
function eTemplate_Callback(hObject, ~, handles)

handles.FitModes{handles.FitMode}.Template = handles.eTemplate.String;
guidata(handles.figure1, handles);

