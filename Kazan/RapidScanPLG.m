function varargout = RapidScanPLG(varargin)
% RAPIDSCANPLG M-file for RapidScanPLG.fig
%      RAPIDSCANPLG, by itself, creates a new RAPIDSCANPLG or raises the existing
%      singleton*.
%
%      H = RAPIDSCANPLG returns the handle to a new RAPIDSCANPLG or the handle to
%      the existing singleton*.
%
%      RAPIDSCANPLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RAPIDSCANPLG.M with the given input arguments.
%
%      RAPIDSCANPLG('Property','Value',...) creates a new RAPIDSCANPLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before RapidScanPLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to RapidScanPLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help RapidScanPLG

% Last Modified by GUIDE v2.5 21-Oct-2023 23:11:20

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @RapidScanPLG_OpeningFcn, ...
  'gui_OutputFcn',  @RapidScanPLG_OutputFcn, ...
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
function RapidScanPLG_OpeningFcn(hObject, ~, handles, varargin)

% Choose default command line output for RapidScanPLG
handles.output = hObject;
[fpath] = fileparts(which('kazan'));
inifilename = [fpath '\kazan.ini'];
handles.ini = inimanage(inifilename);
handles.EnableProcessing = 0;

% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function varargout = RapidScanPLG_OutputFcn(~, ~, handles)
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pbDeconvolute_Callback(~, ~, handles)
ProcessDataMain(handles);

% --------------------------------------------------------------------
function ProcessDataMain(handles)
if get(handles.rb3, 'Value')
  ProcessDataDenverNew(handles)
elseif get(handles.rb2, 'Value')
  ProcessDataDenver(handles)
else
  ShowData(handles);
end

% --------------------------------------------------------------------
function ProcessData(ha)

h = guidata(ha.figure1);
handles = guidata(h.hh);

dsc = safeget(handles.src, 'dsc', []);

y_TD = handles.src.y;

if 1
  y_TD = real(y_TD) - 1i*imag(y_TD);
end

if get(ha.pbFilter, 'Value')
  CutOff = eval(get(ha.eCutOff, 'String'));
  Wn = CutOff*1E6 / (1/2/mean(diff(handles.src.ax.x)));
  if Wn < 0.99
    [b,a]=butter(2,Wn);
    y_TD = filter(b,a,y_TD, [], 1);
  end
end

% determine what modality is used
sampling = mean(diff(handles.src.ax.x));
dPh = eval(get(ha.ecph, 'String'));
dtype = get(ha.pmScanType, 'Value');
if dtype == 1
  % linear scan
  Freq = kvgetvalue(safeget(dsc, 'SCAN_Frequency', '500 Hz'));
  ScanWidth = kvgetvalue(safeget(dsc, 'SCAN_ScanWidth', '500 Hz'));
  
  [x_RS,y_RS]=rs_deconvolve(y_TD, ScanWidth, Freq, sampling);
  
  if get(ha.pbPhase, 'Value')
    phase = str2num(get(ha.ePhase, 'String'));
    y_RS = y_RS*exp(-1i*phase/180*pi);
  end
  
  dPh = eval(get(ha.ecph, 'String'));
  np = size(y_RS, 1);
  y_RS = circshift(y_RS, round(np*dPh/360));
  Rs=y_RS+flipud(y_RS);
  Brange = (max(x_RS) - min(x_RS));
  
  if ~get(ha.pbCyclicPhase, 'Value') && get(ha.pbBaseline, 'Value')
    dB = eval(get(ha.edb, 'String'));
    nH = str2double(get(ha.eNharm, 'String'));
    [y_RS, BG] = rs_baseline_sawtooth(x_RS, y_RS, dB, nH);
    x_RS = x_RS(1:np/2);
    y_RS = y_RS(1:np/2, :);
    BG = BG(1:np/2, :);
  end
  
  handles.out = {};
  handles.out{1}.ax = handles.src.ax;
  handles.out{1}.ax.x = x_RS(:);
  handles.out{1}.y = y_RS;
  handles.out{1}.dsc = dsc;
  
  if get(ha.pbCyclicPhase, 'Value')
    handles.out{2}.ax = handles.src.ax;
    handles.out{2}.ax.x = x_RS(:);
    handles.out{2}.ax.Color = 'g';
    y_fcRS = flipud(y_RS);
    handles.out{2}.y = y_RS+y_fcRS;
    handles.out{2}.dsc = dsc;
    
    handles.out{3}.ax = handles.src.ax;
    handles.out{3}.ax.x = x_RS(:);
    handles.out{3}.ax.Color = 'r';
    handles.out{3}.y = y_RS-y_fcRS;
    handles.out{3}.dsc = dsc;
  elseif  get(ha.pbBaseline, 'Value')
    %   nprj2 = size(Rs, 2);
    %   idx1 = 1:2:nprj2;    % indexes for dn field projections
    %   idx2 = 2:2:nprj2;    % indexes for up field projections
    %   shiftsize = round(np*dB/Brange/2);
    %
    %   R1=Rs;
    %   R1(np/2+1:end,idx1)=Rs(np/2+1:end,idx2); %     step2; rearranging
    %   R1(np/2+1:end,idx2)=Rs(np/2+1:end,idx1); %
    %   a0=R1(:,idx2);
    %   b0=R1(:,idx1);
    %   a=circshift(a0,-shiftsize);       % step 3; circularly shifting
    %   b=circshift(b0,shiftsize);
    
    %   handles.out{2}.ax = handles.src.ax;
    %   handles.out{2}.ax.x = x_RS(:);
    %   handles.out{2}.ax.Color = 'g';
    %   handles.out{2}.y = BG;
    %   handles.out{2}.dsc = dsc;
    %
    %   handles.out{3}.ax = handles.src.ax;
    %   handles.out{3}.ax.x = x_RS(:);
    %   handles.out{3}.ax.Color = 'r';
    %   handles.out{3}.y = a(1:np/2,:)-b(1:np/2,:);
    %   handles.out{3}.dsc = dsc;
  end
else % sinusoidal
  Freq = kvgetvalue(safeget(handles.src.dsc, 'aliases_RSfrequency', 1e3));
  ScanWidth = kvgetvalue(safeget(dsc, 'aliases_RSwidth', '1 G'));
  
  if get(ha.pbPhase, 'Value')
    phase = str2double(get(ha.ePhase, 'String'));
    y_TD = y_TD*exp(-1i*phase/180*pi);
  end
  
  pars.field_scan_phase = dPh;
  pars.display = 'off';
  pars.data_phase = 0;
  
  if get(ha.pmScanDir, 'Value') == 2
    pars.up_down = 'down_up';
  else
    pars.up_down = 'up_down';
  end
  
  auto_phase = get(ha.cbAutoPhase, 'value');
  if auto_phase
    [x_RS,y_RS,pars.field_scan_phase]=rs_sscan_phase(y_TD, ScanWidth, Freq, sampling, pars);
    set(ha.ecph, 'string', sprintf('%5.3f', pars.field_scan_phase));
  else
    pars.N_iter=1;
    [~,y_RS_r]=rs_sdeconvolve(y_TD, ScanWidth, Freq, sampling, pars);
    pars2 = pars;
    pars2.data_phase = pars.data_phase + 90;
    [x_RS,y_RS_i]=rs_sdeconvolve(y_TD, ScanWidth, Freq, sampling, pars2);
    y_RS = y_RS_r + 1i * y_RS_i;
  end
    
  handles.out = {};
  handles.out{1}.ax = handles.src.ax;
  handles.out{1}.ax.x = x_RS(:);
  handles.out{1}.y = y_RS;
  handles.out{1}.dsc = dsc;
  
  if get(ha.pbCyclicPhase, 'Value')
    handles.out{2}.ax = handles.src.ax;
    handles.out{2}.ax.x = x_RS(:);
    handles.out{2}.ax.Color = 'g';
    y_fcRS = flipud(y_RS);
    handles.out{2}.y = y_RS+y_fcRS;
    handles.out{2}.dsc = dsc;
    
    handles.out{3}.ax = handles.src.ax;
    handles.out{3}.ax.x = x_RS(:);
    handles.out{3}.ax.Color = 'r';
    handles.out{3}.y = y_RS-y_fcRS;
    handles.out{3}.dsc = dsc;
  end
end

guidata(h.hh, handles);
[~,name] = fileparts(get(h.hh, 'FileName'));
eval([name '(''SetDataSource'', 2, handles)']);

% --------------------------------------------------------------------
function ProcessDataDenver(ha)

[y_TD, pars, dsc] = get_data(ha);

if get(ha.pbFilter, 'Value')
  CutOff = eval(get(ha.eCutOff, 'String'));
  Wn = CutOff*1E6 / (1/2/mean(pars.sampling));
  if Wn < 0.99
    [b,a]=butter(2,Wn);
    y_TD = filter(b,a,y_TD, [], 1);
  end
end

% determine what modality is used
dPh = eval(get(ha.ecph, 'String'));
dtype = get(ha.pmScanType, 'Value');
if dtype == 1
  % linear scan
  Freq = kvgetvalue(safeget(dsc, 'SCAN_Frequency', '500 Hz'));
  ScanWidth = kvgetvalue(safeget(dsc, 'SCAN_ScanWidth', '500 Hz'));
  
  [x_RS,y_RS]=rs_deconvolve(y_TD, ScanWidth, Freq, sampling);
  
  if get(ha.pbPhase, 'Value')
    phase = str2num(get(ha.ePhase, 'String'));
    y_RS = y_RS*exp(-1i*phase/180*pi);
  end
  
  dPh = eval(get(ha.ecph, 'String'));
  np = size(y_RS, 1);
  y_RS = circshift(y_RS, round(np*dPh/360));
  Rs=y_RS+flipud(y_RS);
  Brange = (max(x_RS) - min(x_RS));
  
  if ~get(ha.pbCyclicPhase, 'Value') && get(ha.pbBaseline, 'Value')
    dB = eval(get(ha.edb, 'String'));
    nH = str2double(get(ha.eNharm, 'String'));
    [y_RS, BG] = rs_baseline_sawtooth(x_RS, y_RS, dB, nH);
    x_RS = x_RS(1:np/2);
    y_RS = y_RS(1:np/2, :);
    BG = BG(1:np/2, :);
  end
  
  handles.out = {};
  handles.out{1}.ax = handles.src.ax;
  handles.out{1}.ax.x = x_RS(:);
  handles.out{1}.y = y_RS;
  handles.out{1}.dsc = dsc;
  
  if get(ha.pbCyclicPhase, 'Value')
    handles.out{2}.ax = handles.src.ax;
    handles.out{2}.ax.x = x_RS(:);
    handles.out{2}.ax.Color = 'g';
    y_fcRS = flipud(y_RS);
    handles.out{2}.y = y_RS+y_fcRS;
    handles.out{2}.dsc = dsc;
    
    handles.out{3}.ax = handles.src.ax;
    handles.out{3}.ax.x = x_RS(:);
    handles.out{3}.ax.Color = 'r';
    handles.out{3}.y = y_RS-y_fcRS;
    handles.out{3}.dsc = dsc;
  elseif  get(ha.pbBaseline, 'Value')
    %   nprj2 = size(Rs, 2);
    %   idx1 = 1:2:nprj2;    % indexes for dn field projections
    %   idx2 = 2:2:nprj2;    % indexes for up field projections
    %   shiftsize = round(np*dB/Brange/2);
    %
    %   R1=Rs;
    %   R1(np/2+1:end,idx1)=Rs(np/2+1:end,idx2); %     step2; rearranging
    %   R1(np/2+1:end,idx2)=Rs(np/2+1:end,idx1); %
    %   a0=R1(:,idx2);
    %   b0=R1(:,idx1);
    %   a=circshift(a0,-shiftsize);       % step 3; circularly shifting
    %   b=circshift(b0,shiftsize);
    
    %   handles.out{2}.ax = handles.src.ax;
    %   handles.out{2}.ax.x = x_RS(:);
    %   handles.out{2}.ax.Color = 'g';
    %   handles.out{2}.y = BG;
    %   handles.out{2}.dsc = dsc;
    %
    %   handles.out{3}.ax = handles.src.ax;
    %   handles.out{3}.ax.x = x_RS(:);
    %   handles.out{3}.ax.Color = 'r';
    %   handles.out{3}.y = a(1:np/2,:)-b(1:np/2,:);
    %   handles.out{3}.dsc = dsc;
  end
else % sinusoidal
  
  if get(ha.pbPhase, 'Value')
    phase = str2double(get(ha.ePhase, 'String'));
    y_TD = y_TD*exp(-1i*phase/180*pi);
  end
  
  pars.field_scan_phase = dPh;
  pars.display = 'off';
  pars.data_phase = 0;
  
  if get(ha.pmScanDir, 'Value') == 2
    pars.up_down = 'down_up';
  else
    pars.up_down = 'up_down';
  end

  auto_phase = get(ha.cbAutoPhase, 'value');
  if auto_phase
    [x_RS,y_RS,pars.field_scan_phase]=rs_sscan_phase(y_TD, pars.ScanWidth, pars.Freq, pars.sampling, pars);
    set(ha.ecph, 'string', sprintf('%5.3f', pars.field_scan_phase));
  else
    pars.N_iter=1;
    [~,y_RS_r]=rs_sdeconvolve(y_TD, pars.ScanWidth, pars.Freq, pars.sampling, pars);
    pars2 = pars;
    pars2.data_phase = pars.data_phase + 90;
    [x_RS,y_RS_i]=rs_sdeconvolve(y_TD, pars.ScanWidth, pars.Freq, pars.sampling, pars2);
    y_RS = y_RS_r + 1i * y_RS_i;
  end
  
  handles = guidata(ha.hh);
  handles.out = {};
  handles.out{1}.ax = [];
  handles.out{1}.ax.x = x_RS(:);
  handles.out{1}.y = y_RS;
  handles.out{1}.dsc = dsc;
  
  if get(ha.pbCyclicPhase, 'Value')
    handles.out{2}.ax = [];
    handles.out{2}.ax.x = x_RS(:);
    handles.out{2}.ax.Color = 'g';
    y_fcRS = flipud(y_RS);
    handles.out{2}.y = y_RS+y_fcRS;
    handles.out{2}.dsc = dsc;
    
    handles.out{3}.ax = handles.src.ax;
    handles.out{3}.ax.x = x_RS(:);
    handles.out{3}.ax.Color = 'r';
    handles.out{3}.y = y_RS-y_fcRS;
    handles.out{3}.dsc = dsc;
  end
end

guidata(ha.hh, handles);
[~,name] = fileparts(get(ha.hh, 'FileName'));
eval([name '(''SetDataSource'', 2, handles)']);

% --------------------------------------------------------------------
function ProcessDataDenverNew(ha)

[y_TD, pars, dsc] = get_data(ha);
% y_TD = -imag(y_TD) + 1i*real(y_TD);

par.hm = pars.ScanWidth; % [G] Peak-to-peak modulation amplitude
par.Vm = pars.Freq;      % [Hz] Modulation frequency
par.rs = y_TD;           % Experimantal Rapid scan signal, must have at least one full cycle
par.dt = pars.sampling;  %  dt -  time base (sampling period) for signal stored in array [s]
par.up = iff(ha.pmScanDir.Value == 1, 1, -1);   % up must be '1' if
                                                % the first half-scan has up direction, '-1' for down-scan.
par.ph = iff(ha.pbPhase.Value, str2double(ha.ePhase.String), 0);  %  RF (MW) phase correction [degrees]                                                   

if  ha.cbAutoPhase.Value
  par.fp = 0;
  par.N_iter = 4;
else
  par.fp = eval(ha.ecph.String);
  par.N_iter = 1;
  % par.fp     First point of the cycle to be deconvolved in units of degrees [0 360]
  %            For experimental data fp has to be adjusted,so that A and B (spectra for up and down scans) coinside.
  % par.bw;    Currently is not used
  %            [Hz] Expected signal BW for pre-filtering, should be slighty
  %            larger than the resonator BW
  % dPh = eval(get(ha.ecph, 'String'));
  % np = size(y_RS, 1);
  % y_RS = circshift(y_RS, round(np*dPh/360));
  % Rs=y_RS+flipud(y_RS);
  % Brange = (max(x_RS) - min(x_RS));
end

par.fwhm = pars.ScanWidth / 25;
% par.fwhm;  [G] Estimated full width at the half height of the narrowest line in the spectrum.
%            It is used for spectrum filtering with a gaussain profile with width =par.fwhm/10

par.msg = 1;  %  '1' to show, '0' switches off warning messages to speed up data processing
par.fig = 0;  % '1' is used for adjustment of par.fp to match up and down spectra;
%            '0' does not refresh the figure and is used to speed up the
%             deconvolution program for automated data processing.

par.method='fast';  % default is 'fast', for very sensetive experiments you may try  'slow'

[h, A, B, fp_corr_total, rs_out, bg_out]=sinDecoBG(par);
% x_RS = linspace(-par.hm/2, par.hm/2, length(A));

if  ha.cbAutoPhase.Value
  ha.ecph.String = num2str(fp_corr_total);
end

handles = guidata(ha.hh);
handles.out = {};
handles.out{1}.ax = [];
handles.out{1}.ax.x = h(:);
handles.out{1}.y = A(:) + B(:);
handles.out{1}.dsc = dsc;

if get(ha.pbCyclicPhase, 'Value')
  handles = guidata(ha.hh);
  handles.out = {};
  handles.out{1}.ax = [];
  handles.out{1}.ax.x = h(:);
  handles.out{1}.y = A(:);
  handles.out{1}.dsc = dsc;
  handles.out{1}.ax.Color = 'r';

  handles.out{2}.ax = [];
  handles.out{2}.ax.x = h(:);
  handles.out{2}.y = B(:);
  handles.out{2}.dsc = dsc;
  handles.out{2}.ax.Color = 'g';
else
  handles = guidata(ha.hh);
  handles.out = {};
  handles.out{1}.ax = [];
  handles.out{1}.ax.x = h(:);
  handles.out{1}.y = A(:) + B(:);
  handles.out{1}.dsc = dsc;
end

guidata(ha.hh, handles);
[~,name] = fileparts(get(ha.hh, 'FileName'));
eval([name '(''SetDataSource'', 2, handles)']);

% --------------------------------------------------------------------
function ShowData(handles)

ha = guidata(handles.hh);
[y, pars, dsc] = get_data(handles);

dt = pars.sampling;
x = (0:size(y,1)-1) * dt;

phase = str2num(get(handles.ePhase, 'String'));
sphase = str2num(get(handles.ecph, 'String'));

dtype = get(handles.pmScanType, 'Value');

if get(handles.pbFilter, 'Value')
  CutOff = eval(get(handles.eCutOff, 'String'));
  Wn = CutOff*1E6 / (1/2/mean(diff(x)));
  if Wn < 0.99
    [b,a]=butter(2,Wn);
    y = filter(b,a,y, [], 1);
  end
end

if dtype == 1
%   Freq = kvgetvalue(safeget(dsc, 'SCAN_Frequency', '500 Hz'));
%   ScanWidth = kvgetvalue(safeget(dsc, 'SCAN_ScanWidth', '500 Hz'));
else

end

% Average projections
Npt  = size(y, 1);           % trace length
Ntr  = size(y, 2);           % number of traces
P  =  1/pars.Freq;           % Scan period
N  = fix(Npt*dt/P);          % number of cycles in the trace

M=2^16;
t = dt * (0:Npt-1);
x_ss = (0:(M-1))'/M * pars.ScanWidth*2;

rsi = zeros(M, Ntr);
for ii=1:Ntr
  for jj=1:N
    pIdx = t >= ((jj-1) * P - 2*dt) & t < (jj * P + 2*dt);
    
    x1 = t(pIdx)';
    x2 = linspace((jj-1) * P, jj * P - dt, M)';
    rsi_trace = interp1(x1,y(pIdx,ii),x2);
    rsi_trace(isnan(rsi_trace)) = 0;
    rsi(:,ii)=rsi(:,ii)+rsi_trace;
  end
end
rsi=rsi / N;

rsi = circshift(rsi, round(sphase/360*M));

if dtype == 2
  rss = zeros(size(rsi));
  x_ss1 = pars.ScanWidth*cos(linspace(0,2*pi, M+1)+pi); x_ss1 = x_ss1(1:end-1);
  x_i   = pars.ScanWidth*linspace(-1,1,M/2+1); x_i = x_i(1:end-1);
  rss(1:M/2,:) = interp1(x_ss1(1:M/2), rsi(1:M/2,:), x_i);
  rss(M/2+1:end,:) = interp1(x_ss1(M/2+1:end), rsi(M/2+1:end,:), -x_i);
  %   figure; plot(x_i, rsi(1:M/2,:), x_i, rss(1:M/2,:));
  rsi = rss;
end

if get(handles.pbPhase, 'Value')
  rsi = rsi*exp(-1i*phase/180*pi);
end

ha.out = {};
ha.out{1}.ax = ha.src.ax;
ha.out{1}.ax.x = x_ss(:);
ha.out{1}.y = rsi;
ha.out{1}.dsc = dsc;

if get(handles.pbCyclicPhase, 'Value')
  ha.out{2}.ax = ha.src.ax;
  ha.out{2}.ax.x = x_ss(:);
  ha.out{2}.ax.Color = 'r';
  ha.out{2}.y = rsi-flipud(rsi);
  ha.out{2}.dsc = dsc;
end

guidata(handles.hh, ha);
[path,name] = fileparts(get(handles.hh, 'FileName'));
eval([name '(''SetDataSource'', 2, ha)']);

% --------------------------------------------------------------------
function pbPhase_Callback(hObject, eventdata, handles)
guidata(handles.figure1, handles);
shift =get(handles.slPhase, 'Value');
set(handles.slPhase, 'Value', 0);
CurVal=str2num(get(handles.ePhase, 'String'));
cc = safeget(handles, 'PhaseFactor', 1.);
Val = constrain_phase(CurVal + shift*cc);
set(handles.ePhase, 'String', num2str(Val))
ProcessDataMain(handles);

% --------------------------------------------------------------------
function ePhase_Callback(hObject, ~, handles)
guidata(handles.figure1, handles);
switch hObject
  case handles.slPhase
    shift =get(handles.slPhase, 'Value');
    set(handles.slPhase, 'Value', 0);
    CurVal=str2num(get(handles.ePhase, 'String'));
    cc = safeget(handles, 'PhaseFactor',1.);
    Val =constrain_phase(CurVal + shift*cc);
    set(handles.ePhase, 'String', num2str(Val))
end
ProcessDataMain(handles);

% --------------------------------------------------------------------
function slPhase_Callback(hObject, ~, handles)
guidata(handles.figure1, handles);
shift = get(handles.slPhase, 'Value');
set(handles.slPhase, 'Value', 0);
CurVal=str2num(get(handles.ePhase, 'String'));
cc = safeget(handles, 'PhaseFactor',1.);
Val = constrain_phase(CurVal + shift*cc);
set(handles.ePhase, 'String', num2str(Val))
ProcessDataMain(handles);

% --------------------------------------------------------------------
function pb01_Callback(hObject, ~, handles)
if get(handles.pb01, 'Value'), handles.PhaseFactor = 0.1; set([handles.pb10,handles.pb0001], 'Value',0);
else handles.PhaseFactor = 1;
end
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function pb10_Callback(hObject, eventdata, handles)
if get(handles.pb10, 'Value'), handles.PhaseFactor = 10; set([handles.pb01,handles.pb0001,handles.pb00001], 'Value',0);
else handles.PhaseFactor = 1;
end
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function pb00001_Callback(hObject, ~, handles)
if get(handles.pb00001, 'Value'), handles.PhaseFactor = 0.001; set([handles.pb01,handles.pb10,handles.pb0001], 'Value',0);
else handles.PhaseFactor = 1; end
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function pb0001_Callback(hObject, ~, handles)
if get(handles.pb0001, 'Value'), handles.PhaseFactor = 0.01; set([handles.pb01,handles.pb10,handles.pb00001], 'Value',0);
else handles.PhaseFactor = 1; end
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function pbBaseline_Callback(~, ~, handles)
ProcessDataMain(handles);

% --------------------------------------------------------------------
function pbCyclicPhase_Callback(~, ~, handles)
ProcessDataMain(handles);

% --------------------------------------------------------------------
function edb_Callback(~, ~, handles)
ProcessDataMain(handles);

% --------------------------------------------------------------------
function eNharm_Callback(~, ~, handles)
ProcessDataMain(handles);

% --------------------------------------------------------------------
function eCutOff_Callback(~, ~, handles)
ProcessDataMain(handles);

% --------------------------------------------------------------------
function pbFilter_Callback(~, ~, handles)
ProcessDataMain(handles);

% --------------------------------------------------------------------
function ecph_Callback(~, ~, handles)
ProcessDataMain(handles);

% --------------------------------------------------------------------
function slScanPhase_Callback(~, ~, handles)
guidata(handles.figure1, handles);
shift = get(handles.slScanPhase, 'Value');
set(handles.slScanPhase, 'Value', 0);
CurVal=str2num(get(handles.ecph, 'String'));
cc = safeget(handles, 'PhaseFactor', 1.);
Val = constrain_phase(CurVal + shift*cc);
set(handles.ecph, 'String', num2str(Val))
ProcessDataMain(handles);

% --------------------------------------------------------------------
function sldB_Callback(~, ~, handles)
guidata(handles.figure1, handles);
shift = get(handles.sldB, 'Value');
set(handles.sldB, 'Value', 0);
CurVal=str2double(get(handles.edb, 'String'));
cc = safeget(handles, 'PhaseFactor', 1.)*0.1;
Val = constrain_phase(CurVal + shift*cc);
set(handles.edb, 'String', num2str(Val))
ProcessDataMain(handles);

% --------------------------------------------------------------------
function pbAutoSearch_Callback(~, ~, handles)

[y_TD, pars] = get_data(handles);

pars.N_iter=1;
pars.display = 'off';
pars.data_phase = 0;


a0 = str2double(handles.ecph.String);
if handles.pb10.Value
  span0 = 360; 
elseif handles.pb01.Value
  span0 = 20; 
elseif handles.pb0001.Value
  span0 = 10; 
elseif handles.pb00001.Value
  span0 = 5; 
else
  span0 = 40; 
end

a1 = str2double(handles.ePhase.String);
for ii=1:3
  a0 = get_max_scan_angle(y_TD*exp(-1i*a1/180*pi), pars, linspace(a0-span0/2, a0+span0/2, 50));
  pars.field_scan_phase = a0;
  span0 = span0 / 5;
  a1 = get_max_data_phase(y_TD, pars);
end

handles.ePhase.String = num2str(a1);
handles.ecph.String = num2str(a0);
fprintf('Optimization is finished.\n');

ProcessDataMain(handles);

% --------------------------------------------------------------------
function ret = get_max_scan_angle(y_TD, pars, angles)
tic
idx = zeros(size(angles));
for ii=1:length(angles)
  pars.field_scan_phase = angles(ii);
  [~,y_RS_r]=rs_sdeconvolve(y_TD, pars.ScanWidth, pars.Freq, pars.sampling, pars);
  ff = sgolayfilt(diff(real(y_RS_r)), 1, 13);
  idx(ii)=max(ff);
end
[~,iii] = max(idx);
ret = angles(iii);
figure(1); clf; plot(angles, idx); hold on; plot(ret,idx(iii),'o')
toc

% --------------------------------------------------------------------
function r = fit(complex_data, phase)
data = real(complex_data * exp(-1i*phase/180*pi));
r    = abs(max(data) + min(data));

function ret = get_max_data_phase(y_TD, pars)
tic
[~,y_RS_r]=rs_sdeconvolve(y_TD, pars.ScanWidth, pars.Freq, pars.sampling, pars);
[~,y_RS_i]=rs_sdeconvolve(y_TD*exp(-1i*pi/2), pars.ScanWidth, pars.Freq, pars.sampling, pars);

y_RS = sgolayfilt(diff(real(y_RS_r)), 1, 21) + ...
  1i*sgolayfilt(diff(real(y_RS_i)), 1, 21);

fun = @(x) fit(y_RS, x);

ret = fminbnd(fun,-180,180);
toc

% --------------------------------------------------------------------
function [y_TD, pars, dsc] = get_data(handles)

h = guidata(handles.figure1);
ha = guidata(h.hh);

dsc = safeget(ha.src, 'dsc', []);

y_TD = ha.src.y;

if isfield(dsc,'aliases_RSfrequency')
  pars.origin = 'specman';
  y_TD = real(y_TD) - 1i*imag(y_TD);
  pars.Freq = kvgetvalue(safeget(dsc, 'aliases_RSfrequency', '500 Hz'));
  pars.ScanWidth = kvgetvalue(safeget(dsc, 'aliases_RSwidth', '500 Hz'));
  pars.sampling = mean(diff(ha.src.ax.x));
else
  pars.origin = 'denver';
  pars.Freq = str2double(safeget(dsc, 'ScanFrequency', '1'))*1E3; % Hz
  pars.ScanWidth = str2double(safeget(dsc, 'ScanWidth', '7'))*10; % G
  pars.sampling = mean(diff(ha.src.ax.x))*1e-9;
end

% --------------------------------------------------------------------
function new_phase = constrain_phase(phase)

if phase > 180, new_phase = phase - 360;
elseif phase < -360, new_phase = phase + 180;
else
  new_phase = phase;
end

% --------------------------------------------------------------------
function pbShowScript_Callback(~, ~, ha)
[y_TD, pars, dsc] = get_data(ha);
fprintf('--------------\n')
fprintf('y_TD = load();\n')

if get(ha.pbFilter, 'Value')
  CutOff = eval(get(ha.eCutOff, 'String'));
  Wn = CutOff*1E6 / (1/2/mean(pars.sampling));
  if Wn < 0.99
    fprintf('[b,a]=butter(2,%5.3f)\n', Wn);
    fprintf('y_TD = filter(b,a,y_TD, [], 1\n');
  end
end

% determine what modality is used
dPh = eval(get(ha.ecph, 'String'));
dtype = get(ha.pmScanType, 'Value');
if dtype == 1
  % linear scan
  Freq = kvgetvalue(safeget(dsc, 'SCAN_Frequency', '500 Hz'));
  ScanWidth = kvgetvalue(safeget(dsc, 'SCAN_ScanWidth', '500 Hz'));
  
  [x_RS,y_RS]=rs_deconvolve(y_TD, ScanWidth, Freq, sampling);
  
  if get(ha.pbPhase, 'Value')
    phase = str2num(get(ha.ePhase, 'String'));
    y_RS = y_RS*exp(-1i*phase/180*pi);
  end
  
  dPh = eval(get(ha.ecph, 'String'));
  np = size(y_RS, 1);
  y_RS = circshift(y_RS, round(np*dPh/360));
  Rs=y_RS+flipud(y_RS);
  Brange = (max(x_RS) - min(x_RS));
  
  if ~get(ha.pbCyclicPhase, 'Value') && get(ha.pbBaseline, 'Value')
    dB = eval(get(ha.edb, 'String'));
    nH = str2double(get(ha.eNharm, 'String'));
    [y_RS, BG] = rs_baseline_sawtooth(x_RS, y_RS, dB, nH);
    x_RS = x_RS(1:np/2);
    y_RS = y_RS(1:np/2, :);
    BG = BG(1:np/2, :);
  end
  
  handles.out = {};
  handles.out{1}.ax = handles.src.ax;
  handles.out{1}.ax.x = x_RS(:);
  handles.out{1}.y = y_RS;
  handles.out{1}.dsc = dsc;
  
  if get(ha.pbCyclicPhase, 'Value')
    handles.out{2}.ax = handles.src.ax;
    handles.out{2}.ax.x = x_RS(:);
    handles.out{2}.ax.Color = 'g';
    y_fcRS = flipud(y_RS);
    handles.out{2}.y = y_RS+y_fcRS;
    handles.out{2}.dsc = dsc;
    
    handles.out{3}.ax = handles.src.ax;
    handles.out{3}.ax.x = x_RS(:);
    handles.out{3}.ax.Color = 'r';
    handles.out{3}.y = y_RS-y_fcRS;
    handles.out{3}.dsc = dsc;
  elseif  get(ha.pbBaseline, 'Value')
    %   nprj2 = size(Rs, 2);
    %   idx1 = 1:2:nprj2;    % indexes for dn field projections
    %   idx2 = 2:2:nprj2;    % indexes for up field projections
    %   shiftsize = round(np*dB/Brange/2);
    %
    %   R1=Rs;
    %   R1(np/2+1:end,idx1)=Rs(np/2+1:end,idx2); %     step2; rearranging
    %   R1(np/2+1:end,idx2)=Rs(np/2+1:end,idx1); %
    %   a0=R1(:,idx2);
    %   b0=R1(:,idx1);
    %   a=circshift(a0,-shiftsize);       % step 3; circularly shifting
    %   b=circshift(b0,shiftsize);
    
    %   handles.out{2}.ax = handles.src.ax;
    %   handles.out{2}.ax.x = x_RS(:);
    %   handles.out{2}.ax.Color = 'g';
    %   handles.out{2}.y = BG;
    %   handles.out{2}.dsc = dsc;
    %
    %   handles.out{3}.ax = handles.src.ax;
    %   handles.out{3}.ax.x = x_RS(:);
    %   handles.out{3}.ax.Color = 'r';
    %   handles.out{3}.y = a(1:np/2,:)-b(1:np/2,:);
    %   handles.out{3}.dsc = dsc;
  end
else % sinusoidal
  
  if get(ha.pbPhase, 'Value')
    phase = str2double(get(ha.ePhase, 'String'));
    fprintf('y_TD = y_TD*exp(-1i*%5.3f/180*pi);\n', phase);
  end
  
  fprintf('pars.field_scan_phase = %5.3f;\n', dPh);
  fprintf('pars.display = ''off'';\n');
  fprintf('pars.data_phase = 0;\n');

  fprintf('pars.ScanWidth = %5.3f;\n', pars.ScanWidth);
  fprintf('pars.Freq = %6.1f;\n', pars.Freq);
  fprintf('pars.sampling = %6.1fE-9;\n', pars.sampling*1e9);


  if get(ha.pmScanDir, 'Value') == 2
    fprintf('pars.up_down = ''down_up'';\n');
  else
    fprintf('pars.up_down = ''up_down'';\n');
  end

  fprintf('pars.N_iter=1;\n');
  fprintf('[x_RS,y_RS_r]=rs_sdeconvolve(y_TD, pars.ScanWidth, pars.Freq, pars.sampling, pars);\n');
end
