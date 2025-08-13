function varargout = FindPhaseFig(varargin)
% FINDPHASEFIG M-file for FindPhaseFig.fig
%      FINDPHASEFIG, by itself, creates a new FINDPHASEFIG or raises the existing
%      singleton*.
%
%      H = FINDPHASEFIG returns the handle to a new FINDPHASEFIG or the handle to
%      the existing singleton*.
%
%      FINDPHASEFIG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in FINDPHASEFIG.M with the given input arguments.
%
%      FINDPHASEFIG('Property','Value',...) creates a new FINDPHASEFIG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before FindPhaseFig_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to FindPhaseFig_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help FindPhaseFig

% Last Modified by GUIDE v2.5 29-Jan-2016 16:40:16

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @FindPhaseFig_OpeningFcn, ...
  'gui_OutputFcn',  @FindPhaseFig_OutputFcn, ...
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
function FindPhaseFig_OpeningFcn(hObject, eventdata, handles, varargin)

% Choose default command line output for FindPhaseFig
handles.output = hObject;
handles.data = varargin{1};
handles.EnableProcessing = 0;

set(handles.edb, 'string', num2str(handles.data.FieldSweep));

% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function varargout = FindPhaseFig_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pbDeconvolute_Callback(hObject, eventdata, handles)
if get(handles.rb2, 'Value')
  ProcessData(handles)
else
  ShowData(handles);
end

% --------------------------------------------------------------------
function ProcessData(ha)

y_TD = ha.data.y;

if get(ha.pbFilter, 'Value')
  CutOff = eval(get(ha.eCutOff, 'String'));
  Wn = CutOff*1E6 / (1/2/mean(diff(handles.src.ax.x)));
  if Wn < 0.99
    [b,a]=butter(2,Wn);
    y_TD = filter(b,a,y_TD, [], 1);
  end
end

% determine what modality is used
sampling = ha.data.sampling;
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
  Freq = ha.data.RSfrequency;
  ScanWidth = ha.data.FieldSweep;
  
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
  
  cla(ha.axes1);
  plot(ha.axes1, x_RS(:), real(y_RS)); hold on
%   plot(ha.axes1, x_RS(:), [0;diff(real(y_RS))]);
  axis(ha.axes1, 'tight'); grid(ha.axes1, 'on');
  
  maxy = max(real(y_RS));
  miny = min(real(y_RS));
  [pks,locs] = findpeaks(real(y_RS),x_RS(:),'MinPeakProminence', 0.2*(maxy-miny));
  
  if length(pks) < 7
    str = {'data'};
    for ii=1:length(locs)
      str{end+1} = sprintf('%3.2f', locs(ii));
      plot(ha.axes1, locs(ii), pks(ii), 'o');
    end
    legend(str);
  end
  
  if get(ha.pbCyclicPhase, 'Value')
%     handles.out{2}.ax = handles.src.ax;
%     handles.out{2}.ax.x = x_RS(:);
%     handles.out{2}.ax.Color = 'g';
%     y_fcRS = flipud(y_RS);
%     handles.out{2}.y = y_RS+y_fcRS;
%     handles.out{2}.dsc = dsc;
%     
%     handles.out{3}.ax = handles.src.ax;
%     handles.out{3}.ax.x = x_RS(:);
%     handles.out{3}.ax.Color = 'r';
%     handles.out{3}.y = y_RS-y_fcRS;
%     handles.out{3}.dsc = dsc;
  end
end

% --------------------------------------------------------------------
function pbPhase_Callback(hObject, eventdata, handles)
guidata(handles.figure1, handles);
shift =get(handles.slPhase, 'Value');
set(handles.slPhase, 'Value', 0);
CurVal=str2num(get(handles.ePhase, 'String'));
cc = safeget(handles, 'PhaseFactor', 1.);
Val = max(min(CurVal + shift*cc, 180), -180);
set(handles.ePhase, 'String', num2str(Val))
if get(handles.rb2, 'Value')
  ProcessData(handles)
else
  ShowData(handles);
end

% --------------------------------------------------------------------
function ePhase_Callback(hObject, eventdata, handles)
guidata(handles.figure1, handles);
switch hObject
  case handles.slPhase
    shift =get(handles.slPhase, 'Value');
    set(handles.slPhase, 'Value', 0);
    CurVal=str2num(get(handles.ePhase, 'String'));
    cc = safeget(handles, 'PhaseFactor',1.);
    Val = max(min(CurVal + shift*cc, 180), -180);
    set(handles.ePhase, 'String', num2str(Val))
end
if get(handles.rb2, 'Value')
  ProcessData(handles)
else
  ShowData(handles);
end

% --------------------------------------------------------------------
function slPhase_Callback(hObject, eventdata, handles)
guidata(handles.figure1, handles);
shift = get(handles.slPhase, 'Value');
set(handles.slPhase, 'Value', 0);
CurVal=str2num(get(handles.ePhase, 'String'));
cc = safeget(handles, 'PhaseFactor',1.);
Val = max(min(CurVal + shift*cc, 180), -180);
set(handles.ePhase, 'String', num2str(Val))
if get(handles.rb2, 'Value')
  ProcessData(handles)
else
  ShowData(handles);
end

% --------------------------------------------------------------------
function pb01_Callback(hObject, eventdata, handles)
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

% --- Executes on button press in pb0001.
function pb00001_Callback(hObject, eventdata, handles)
if get(handles.pb00001, 'Value'), handles.PhaseFactor = 0.001; set([handles.pb01,handles.pb10,handles.pb0001], 'Value',0);
else handles.PhaseFactor = 1; end
guidata(handles.figure1, handles);

% --- Executes on button press in pb0001.
function pb0001_Callback(hObject, eventdata, handles)
if get(handles.pb0001, 'Value'), handles.PhaseFactor = 0.01; set([handles.pb01,handles.pb10,handles.pb00001], 'Value',0);
else handles.PhaseFactor = 1; end
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function pbBaseline_Callback(hObject, eventdata, handles)
if get(handles.rb2, 'Value')
  ProcessData(handles)
else
  ShowData(handles);
end

% --------------------------------------------------------------------
function pbCyclicPhase_Callback(hObject, eventdata, handles)
if get(handles.rb2, 'Value')
  ProcessData(handles)
else
  ShowData(handles);
end

% --------------------------------------------------------------------
function edb_Callback(hObject, eventdata, handles)
if get(handles.rb2, 'Value')
  ProcessData(handles)
else
  ShowData(handles);
end

% --------------------------------------------------------------------
function eNharm_Callback(hObject, eventdata, handles)
if get(handles.rb2, 'Value')
  ProcessData(handles)
else
  ShowData(handles);
end

% --------------------------------------------------------------------
function eCutOff_Callback(hObject, eventdata, handles)
if get(handles.rb2, 'Value')
  ProcessData(handles)
else
  ShowData(handles);
end

% --------------------------------------------------------------------
function pbFilter_Callback(hObject, eventdata, handles)
if get(handles.rb2, 'Value')
  ProcessData(handles)
else
  ShowData(handles);
end

% --------------------------------------------------------------------
function ecph_Callback(hObject, eventdata, handles)
if get(handles.rb2, 'Value')
  ProcessData(handles)
else
  ShowData(handles);
end

% --------------------------------------------------------------------
function slScanPhase_Callback(hObject, eventdata, handles)
guidata(handles.figure1, handles);
shift = get(handles.slScanPhase, 'Value');
set(handles.slScanPhase, 'Value', 0);
CurVal=str2num(get(handles.ecph, 'String'));
cc = safeget(handles, 'PhaseFactor', 1.);
Val = max(min(CurVal + shift*cc, 180), -180);
set(handles.ecph, 'String', num2str(Val))
if get(handles.rb2, 'Value')
  ProcessData(handles)
else
  ShowData(handles);
end

% --------------------------------------------------------------------
function sldB_Callback(hObject, eventdata, handles)
guidata(handles.figure1, handles);
shift = get(handles.sldB, 'Value');
set(handles.sldB, 'Value', 0);
CurVal=str2num(get(handles.edb, 'String'));
cc = safeget(handles, 'PhaseFactor', 1.)*0.1;
Val = max(min(CurVal + shift*cc, 180), -180);
set(handles.edb, 'String', num2str(Val))
if get(handles.rb2, 'Value')
  ProcessData(handles)
else
  ShowData(handles);
end


