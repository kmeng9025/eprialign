function varargout = FitGUI(varargin)
% FITGUI M-file for FitGUI.fig
%      FITGUI, by itself, creates a new FITGUI or raises the existing
%      singleton*.
%
%      H = FITGUI returns the handle to a new FITGUI or the handle to
%      the existing singleton*.
%
%      FITGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in FITGUI.M with the given input arguments.
%
%      FITGUI('Property','Value',...) creates a new FITGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before FitGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to FitGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help FitGUI

% Last Modified by GUIDE v2.5 27-Apr-2009 11:52:17

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @FitGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @FitGUI_OutputFcn, ...
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
function FitGUI_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;
set(handles.slSlice,'CreateFcn','');

set(handles.mOptions, 'Callback', '');
hh = [handles.mOptionsDisplayData, handles.mOptionsDisplayFit, handles.mOptionsDisplayDifference];
set(hh, 'Callback', ...
  @(hObject,eventdata)FitGUI('mOptionsDisplay_Callback',hObject,eventdata,guidata(hObject)))

handles.PG = zeros(9, 1);
handles.PG_err = zeros(9, 1);
handles.PG_dsc{1}='x_ovr';
handles.PG_dsc{2}='R2 [G]';
handles.PG_dsc{3}='Mod omega [G]';
handles.PG_dsc{4}='Mod amp [G]';
handles.PG_dsc{5}='Obs amp';
handles.PG_dsc{6}='R1 [G]';
handles.PG_dsc{7}='Phase(A/D)';
handles.PG_dsc{8}='Phase(Zeeman)';
handles.PG_dsc{9}='Gaussian sigma';

handles.xData = [];
handles.yData = [];
handles.Fit = [];

handles.other.modcal    = 0.2311;
handles.other.threshold = 0.15;


handles.options.DisplayData = 1;
handles.options.DisplayFit = 1;
handles.options.DisplayDifference = 0;
mOptionsDisplay_Callback(0,[],handles);

% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function varargout = FitGUI_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

% --------------------------------------------------------------------
function figure1_ResizeFcn(hObject, eventdata, handles)
if isfield(handles, 'figure1')
  p = get(handles.figure1, 'Position'); p(1:2) = 0;
  set(handles.axes1, 'Position', p + [35, 30, -80, -40]);
  set(handles.slSlice, 'Position', [p(3)-30, 39, 25, p(4)-70]);
  set(handles.eSlice, 'Position', [p(3)-40, 5, 35, 25]);
  set(handles.cbAll, 'Position', [p(3)-35, p(4)-25, 35, 20]);
end

% --------------------------------------------------------------------
function Plot(handles)
Slice = fix(get(handles.slSlice, 'Value'));
hold(handles.axes1, 'off');
if ~isempty(handles.yData) && handles.options.DisplayData
  if get(handles.cbAll, 'Value')
    plot(handles.xData(:,1), handles.yData, 'Parent', handles.axes1)
  else
    plot(handles.xData(:,Slice), handles.yData(:,Slice), 'Parent', handles.axes1)
  end
else
  delete(get(handles.axes1, 'Children'));
end

if Slice <= size(handles.Fit, 2) || get(handles.cbAll, 'Value')
  hold(handles.axes1, 'on');
  if ~isempty(handles.Fit) && handles.options.DisplayFit
    if get(handles.cbAll, 'Value')
      plot(handles.xData(:,1), handles.Fit, 'r', 'Parent', handles.axes1)
    else
      plot(handles.xData(:,Slice), handles.Fit(:,Slice), 'r', 'Parent', handles.axes1)
    end
  end

  if ~isempty(handles.Fit) && handles.options.DisplayDifference
    if get(handles.cbAll, 'Value')
      plot(handles.xData(:,1), handles.Fit-handles.yData, 'm', 'Parent', handles.axes1)
    else
      plot(handles.xData(:,Slice), handles.Fit(:,Slice)-handles.yData(:,Slice), 'm', 'Parent', handles.axes1)
    end
  end
end
axis tight

% --------------------------------------------------------------------
function slSlice_Callback(hObject, eventdata, handles)

Slice = fix(get(handles.slSlice, 'Value')); 
set(handles.eSlice, 'String', num2str(Slice)); 
Plot(handles);

% --------------------------------------------------------------------
function UpdateDataControls(handles)

MaxSlice = size(handles.yData, 2);

SliderValue = get(handles.slSlice, 'Value');
if MaxSlice == 1, MaxSlice = 1.01; end
if SliderValue > MaxSlice, SliderValue = MaxSlice; end

SliderValue = max(1, SliderValue);
SliderValue = min(MaxSlice, SliderValue);
set(handles.slSlice, 'Max', MaxSlice, 'Value', SliderValue); 
set(handles.slSlice, 'SliderStep', [min(1/(MaxSlice-1), 1), .1]); 

% --------------------------------------------------------------------
function eSlice_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function cbAll_Callback(hObject, eventdata, handles)
Plot(handles)

% --------------------------------------------------------------------
function SetCaption(handles)

str= 'FitGUI ';

if isfield(handles, 'Model')
  str = [str, ' - Model:', handles.Model.Name];
end

set(handles.figure1, 'Name', str)
 
% --------------------------------------------------------------------
function mPars_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function mParsLoadModel_Callback(hObject, eventdata, handles)

str = cw_shf_model();

[model,isOk] = listdlg('PromptString','Select a file:',...
  'SelectionMode','single','ListSize',[160, 250], ...
  'ListString',str);

if isOk
  handles.Model = cw_shf_model(str{model});
  guidata(hObject, handles);
  GenerateParameters(handles);
end
SetCaption(handles)

% --------------------------------------------------------------------
function slSlice_CreateFcn(hObject, eventdata, handles)

% --------------------------------------------------------------------
function mShowCurrent_Callback(hObject, eventdata, handles)

if ~isfield(handles, 'Model')
  disp('Model is not set-up yet.'); return;
end

if ~isfield(handles, 'yData')
  disp('Data are not loaded yet.'); return;
end

SliderValue = fix(get(handles.slSlice, 'Value'));

pars.N = length(handles.Model.Nucs);
pars.idx = [];

pars.x1 = handles.xData(:, SliderValue)-mean(handles.xData(:, SliderValue));

pars.PG = handles.PG(:,SliderValue);
handles.Fit(:,SliderValue) = full_fit([], pars)*4/pars.PG(4);
scale = (handles.Fit(:,SliderValue)'*handles.yData(:,SliderValue))/...
  (handles.Fit(:,SliderValue)'*handles.Fit(:,SliderValue));
handles.Fit(:,SliderValue) = scale * handles.Fit(:,SliderValue);

guidata(hObject, handles);
Plot(handles);

% --------------------------------------------------------------------
function SpSys = GetSpinSystem(handles)
SpSys = [];
if isfield(handles, 'Model')
  SpSys = zeros(length(handles.Model.Nucs), 4);
  for ii=1:length(handles.Model.Nucs)
    SpSys(ii,1) = handles.Model.Nucs{ii}.HF;
    SpSys(ii,2) = handles.Model.Nucs{ii}.Abundance;
    SpSys(ii,3) = handles.Model.Nucs{ii}.Spin; 
    SpSys(ii,4) = handles.Model.Nucs{ii}.EqNucs; 
  end
end

% --------------------------------------------------------------------
function GenerateParameters(handles)

if isfield(handles, 'Model'), N = length(handles.Model.Nucs); else N = 0; end

M = size(handles.yData, 2);
if M < 1, M = 1; end

handles.PG = zeros(4*N + 9, M);
handles.PG_err = zeros(4*N + 9, M);
handles.PG_float = false(4*N + 9);
handles.PG_ub = zeros(4*N + 9, 1);
handles.PG_lb = zeros(4*N + 9, 1);

handles.PG_lb(1:8) = [-2 0.01 0. 0. 0 0 -25*pi/180 -25*pi/180];
handles.PG_ub(1:8) = [2 0.2 1 1 0 0 25*pi/180 25*pi/180];

handles.PG_dsc{1}='x_ovr';
handles.PG_dsc{2}='R2 [G]';
handles.PG_dsc{3}='Mod omega [G]';
handles.PG_dsc{4}='Mod amp [G]';
handles.PG_dsc{5}='Obs amp';
handles.PG_dsc{6}='R1 [G]';
handles.PG_dsc{7}='Phase(A/D)';
handles.PG_dsc{8}='Phase(Zeeman)';

for ii=1:N
  handles.PG_dsc{4*(ii-1)+9}=[handles.Model.Nucs{ii}.Name,' Hyperfine [G]'];
  handles.PG_dsc{4*(ii-1)+10}=[handles.Model.Nucs{ii}.Name,' Abundance [%]'];
  handles.PG_dsc{4*(ii-1)+11}=[handles.Model.Nucs{ii}.Name,' Spin [h]'];
  handles.PG_dsc{4*(ii-1)+12}=[handles.Model.Nucs{ii}.Name,' Equiv Nucs []'];
end

handles.PG(1,:) = 0.0;
handles.PG(2,:) = 0.06;
handles.PG(3,:) = safeget(handles, 'ModOmega', 0.0018);
handles.PG(4,:) = safeget(handles, 'ModAmp', 0.1);

for jj=1:M
  for ii=1:N
    handles.PG(4*(ii-1)+9, jj)=handles.Model.Nucs{ii}.HF;
    handles.PG(4*(ii-1)+10, jj)=handles.Model.Nucs{ii}.Abundance;
    handles.PG(4*(ii-1)+11, jj)=handles.Model.Nucs{ii}.Spin;
    handles.PG(4*(ii-1)+12, jj)=handles.Model.Nucs{ii}.EqNucs;
  end
end

handles.PG_dsc{4*N + 9}='Gaussian sigma';
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function mOptionsDisplay_Callback(hObject, eventdata, handles)
st = {'off','on'};
switch hObject
  case handles.mOptionsDisplayData
    handles.options.DisplayData = ~handles.options.DisplayData;
  case handles.mOptionsDisplayFit 
    handles.options.DisplayFit = ~handles.options.DisplayFit;
  case handles.mOptionsDisplayDifference 
    handles.options.DisplayDifference = ~handles.options.DisplayDifference;
end
set(handles.mOptionsDisplayData, 'Checked', st{handles.options.DisplayData+1});
set(handles.mOptionsDisplayFit, 'Checked', st{handles.options.DisplayFit+1});
set(handles.mOptionsDisplayDifference, 'Checked', st{handles.options.DisplayDifference+1});
if hObject~=0
  guidata(handles.figure1, handles);
  Plot(handles);
end

% --------------------------------------------------------------------
function mParsParameters_Callback(hObject, eventdata, handles)

if isfield(handles, 'Model'), N = length(handles.Model.Nucs); else N = 0; end

M = size(handles.yData, 2);
if M < 1, M = 1; end

d = cell(2*N+9, 1);

pp = 1:8;
for ii=1:N, pp=[pp, 4*(ii-1)+9, 4*(ii-1)+ 10]; end
pp = [pp, 4*N+9];
  
for jj=1:M
  for ii=1:length(pp)
    d{ii,1} = handles.PG_dsc{pp(ii)};
    d{ii,2} = handles.PG(pp(ii), jj);
    d{ii,3} = handles.PG_float(pp(ii));
    d{ii,4} = false;
  end
  dset{jj}=d;
end

pars.Set = fix(get(handles.slSlice, 'Value'));
pars.Set4All = [0, 0, 1, 1];
pars.SetRow4All = [0, 0, 0, 1];
pars.ColumnName = {'Parameter', 'Value', 'Float', 'Set All'};
pars.ColumnFormat = {'char', 'numeric', 'logical'};
pars.ColumnEditable = [false, true, true, true];
pars.ColumnWidth = {0, 55, 40, 40};
[dset, isOk]=EditParametersGUI(dset, pars);

if isOk
  d = dset{1}; for ii=1:length(pp), handles.PG_float(pp(ii))=d{ii,3}; end
  for jj=1:M
    d = dset{jj};
    for ii=1:length(pp), handles.PG(pp(ii), jj)=d{ii,2}; end
  end
  guidata(handles.figure1, handles);
end

% --------------------------------------------------------------------
function mParsEditModel_Callback(hObject, eventdata, handles)

if ~isfield(handles, 'Model'), return; end
N = length(handles.Model.Nucs);
  
dset = cell(5,N);
for ii=1:N
  d{1,1}='Name';
  d{2,1}='Hypefine';
  d{3,1}='Spin';
  d{4,1}='Equiv Nucs';
  d{5,1}='Abundance';

  d{1,2}=handles.Model.Nucs{ii}.Name;
  d{2,2}=handles.Model.Nucs{ii}.HF;
  d{3,2}=handles.Model.Nucs{ii}.Spin;
  d{4,2}=handles.Model.Nucs{ii}.EqNucs;
  d{5,2}=handles.Model.Nucs{ii}.Abundance;
  dset{ii}=d;
end

pars.ColumnName = {'Parameter', 'Value'};
pars.ColumnFormat = {'char', 'numeric'};
pars.ColumnEditable = [false, true, true];
pars.ColumnWidth = {0, 60};
[dset, isOk]=EditParametersGUI(dset, pars);

if isOk
  for ii=1:length(handles.Model.Nucs)
    d = dset{ii};
    handles.Model.Nucs{ii}.Name = d{1,2};
    handles.Model.Nucs{ii}.HF = d{2,2};
    handles.Model.Nucs{ii}.Spin = d{3,2};
    handles.Model.Nucs{ii}.EqNucs = d{4,2};
    handles.Model.Nucs{ii}.Abundance = d{5,2};
  end
  guidata(handles.figure1, handles);
end

% --------------------------------------------------------------------
function mParsBoundaries_Callback(hObject, eventdata, handles)
disp('Later');

% --------------------------------------------------------------------
function mShow_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function mFileSavePars_Callback(hObject, eventdata, handles)

oldpath = safeget(handles, 'PathName', 'c:\Users\Boris\Documents\MATLAB\');

if isfield(handles, 'Model'), N = length(handles.Model.Nucs); else N = 0; end

pp = 1:8;
for ii=1:N, pp=[pp, 4*(ii-1)+9, 4*(ii-1)+10]; end
pp = [pp, 4*N+9];

pp_ext = [];
for ii=1:N, pp_ext=[pp_ext, 4*(ii-1)+11, 4*(ii-1)+12]; end

s.P_text = handles.PG_dsc{pp(1)};
for ii=2:length(pp)
 s.P_text  = str2mat(s.P_text, handles.PG_dsc{pp(ii)});
end
  
s.P = handles.PG(pp,:);
s.P_ext = handles.PG(pp_ext,1);
s.P_errs = handles.PG_err(pp,:); 

s.float = handles.PG_float(pp');
s.pars_out = handles.other.pars_out;
if isfield(handles.other, 'immask'), s.immask = handles.other.immask; end
if isfield(handles.other, 'threshold'), s.threshold = handles.other.threshold; end

[filename, pathname] = uiputfile('*.mat', 'Pick a MAT-file', oldpath);
if ~(isequal(filename,0) || isequal(pathname,0))
  handles.other.save_fname = fullfile(pathname, filename);
  s.xtra_info =  cw_xtra_info(handles.xData, handles.yData, 8, handles.other.xtra_info);
  s.cpv = getcpv(handles.other);
  s.cpv_text = getcpv_text;
  save(fullfile(pathname, filename), '-struct', 's')
  disp(['Image ', fullfile(pathname,filename), ' is saved.']);
end

% --------------------------------------------------------------------
function mFitReport_Callback(hObject, eventdata, handles)
if isfield(handles, 'Model'), N = length(handles.Model.Nucs); else N = 0; end

M = size(handles.yData, 2);
if M < 1, M = 1; end

d = cell(2*N+9, 1);

pp = 1:8;
for ii=1:N, pp=[pp, 4*(ii-1)+9, 4*(ii-1)+ 10]; end
pp = [pp, 4*N+9];
  
for jj=1:M
  for ii=1:length(pp)
    d{ii,1} = handles.PG_dsc{pp(ii)};
    d{ii,2} = handles.PG(pp(ii), jj);
    d{ii,3} = handles.PG_err(pp(ii), jj);
  end
  dset{jj}=d;
end

pars.Set = fix(get(handles.slSlice, 'Value'));
pars.ColumnName = {'Parameter', 'Value', 'Error'};
pars.ColumnFormat = {'char', 'numeric', 'numeric'};
pars.ColumnEditable = [false, false, false];
pars.ColumnWidth = {0, 60, 60};
[dset, isOk]=EditParametersGUI(dset, pars);

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% ------- F I T T I N G   C A L L B A C K S --------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

% --------------------------------------------------------------------
function mFitEstimate_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'Model')
  disp('Model is not set-up yet.'); return;
end

if ~isfield(handles, 'yData')
  disp('Data are not loaded yet.'); return;
end

if ~any(handles.PG_float)
  pars.idx = [1, 2, 7];
  handles.PG_float(:) = false;
  handles.PG_float(pars.idx) = true;
else
  pars.idx = find(handles.PG_float);
end

pars.N = length(handles.Model.Nucs);

Spin_system = reshape(handles.PG(9:end-1, 1), [pars.N, 4])';
[pars.y_pat, del_pat] = cw_shf(Spin_system);

tic
n = size(handles.yData, 2);

% find spectra with high intensity
p2p = max(handles.yData, [], 1) - min(handles.yData, [], 1);
[vals, scale] = hist(p2p,100); vals = cumsum(vals/sum(vals));
min_test = scale(find(vals > 0.6, 1, 'first'));
max_test = scale(find(vals < 0.7, 1, 'last'));
idx = p2p > min_test & p2p < max_test;
if ~any(idx)
  idx = rand(1, length(p2p)) > 0.75;
end
idx1 = find(idx);
idx2 = ~idx;

h = waitbar(0,'Estimating constants...','Name','Fitting is in progress');
for ii=idx1
  %   x = handles.PG(pars.idx, ii);
  pars.x1 = handles.xData(:, ii)-mean(handles.xData(:, ii));
  pars.x_pat = (pars.x1(1)-del_pat:del_pat:pars.x1(end)+del_pat)';
  
  pars.PG  = handles.PG(:, ii);
  pars.yy = handles.yData(:,ii);
  % Robinson minimization
  [xx, xx_err]=cw_grad_min(@simple_fit, pars);
  handles.PG(pars.idx, ii) = xx;
  handles.PG_err(pars.idx, ii) = xx_err;
  handles.Fit(:,ii) = full_fit(xx, pars)*4/pars.PG(4);
  scale = (pars.yy'*handles.Fit(:,ii))/(handles.Fit(:,ii)'*handles.Fit(:,ii));
  % end of Robinson specific code
  handles.Fit(:,ii) = scale * handles.Fit(:,ii);
  handles.other.xtra_info(2,ii) =  abs(scale*handles.Model.n_hf);
  waitbar(ii/n, h);
end
delete(h);

% set estimated values
for ii=1:length(pars.idx)
  handles.PG(pars.idx(ii),idx2) = median(handles.PG(pars.idx(ii),idx1));
end

guidata(hObject, handles);
Plot(handles);

% --------------------------------------------------------------------
function mFitFixedModel_Callback(hObject, eventdata, handles)

if ~isfield(handles, 'Model')
  disp('Model is not set-up yet.'); return;
end

if ~isfield(handles, 'yData')
  disp('Data are not loaded yet.'); return;
end

if ~any(handles.PG_float)
  pars.idx = [1, 2, 7];
  handles.PG_float(:) = false;
  handles.PG_float(pars.idx) = true;
else
  pars.idx = find(handles.PG_float);
end

pars.N = length(handles.Model.Nucs);

Spin_system = reshape(handles.PG(9:end-1, 1), [pars.N, 4])';
[pars.y_pat, del_pat] = cw_shf(Spin_system);

tic
n = size(handles.yData, 2);

estimate_first = 1;

if estimate_first
  % find spectra with high intensity
  p2p = max(handles.yData, [], 1) - min(handles.yData, [], 1);
  [vals, scale] = hist(p2p,30); vals = cumsum(vals/sum(vals));
  min_test = scale(find(vals > 0.6, 1, 'first'));
  max_test = scale(find(vals < 0.7, 1, 'last'));
  idx = p2p > min_test & p2p < max_test;
  if ~any(idx)
    idx = rand(1, length(p2p)) > 0.75;
  end
  idx1 = find(idx);
  idx2 = find(~idx);
  
  h = waitbar(0,'Estimating constants...','Name','Fitting is in progress');
  for ii=idx1
    %   x = handles.PG(pars.idx, ii);
    pars.x1 = handles.xData(:, ii)-mean(handles.xData(:, ii));
    pars.x_pat = (pars.x1(1)-del_pat:del_pat:pars.x1(end)+del_pat)';
    
    pars.PG  = handles.PG(:, ii);
    pars.yy = handles.yData(:,ii);
    % Robinson minimization
    [xx, xx_err]=cw_grad_min(@simple_fit, pars);
    handles.PG(pars.idx, ii) = xx;
    handles.PG_err(pars.idx, ii) = xx_err;
    handles.Fit(:,ii) = full_fit(xx, pars)*4/pars.PG(4);
    scale = (pars.yy'*handles.Fit(:,ii))/(handles.Fit(:,ii)'*handles.Fit(:,ii));
    % end of Robinson specific code
    handles.Fit(:,ii) = scale * handles.Fit(:,ii);
    handles.other.xtra_info(2,ii) =  abs(scale*handles.Model.n_hf);
    waitbar(ii/n, h);
  end
  delete(h);
  
  % set estimated values
  for ii=1:length(pars.idx)
    handles.PG(pars.idx(ii),idx2) = median(handles.PG(pars.idx(ii),idx1));
  end
else
  idx2 = 1:n;
end

h = waitbar(0,'Please wait...','Name','Fitting is in progress');
for ii=idx2
%   x = handles.PG(pars.idx, ii);
  pars.x1 = handles.xData(:, ii)-mean(handles.xData(:, ii));
  pars.x_pat = (pars.x1(1)-del_pat:del_pat:pars.x1(end)+del_pat)';

  pars.PG  = handles.PG(:, ii);
  pars.yy = handles.yData(:,ii);
% MATLAB minimization
%   x = [1;handles.PG_ub(pars.idx)];
% %   x = handles.PG(pars.idx);
%   xx = fmincon(@err_simple_fit,x,[],[],[],[],handles.PG_lb(pars.idx),handles.PG_ub(pars.idx),[],...
%     optimset('MaxFunEvals',1000,'Algorithm','active-set','Display','off', 'TolFun', 1e-8),...
%     pars);
%   xx_err = 0;
%   handles.PG(pars.idx, ii) = xx(2:end);
%   handles.PG_err(pars.idx, ii) = xx_err;
%   handles.Fit(:,ii) = full_fit(xx(2:end), pars)*4/pars.PG(4);
%   scale = x(1);
% end of MATLAB specific code  
% Robinson minimization  
  [xx, xx_err]=cw_grad_min(@simple_fit, pars);
  handles.PG(pars.idx, ii) = xx;
  handles.PG_err(pars.idx, ii) = xx_err;
  handles.Fit(:,ii) = full_fit(xx, pars)*4/pars.PG(4);
  scale = (pars.yy'*handles.Fit(:,ii))/(handles.Fit(:,ii)'*handles.Fit(:,ii));
% end of Robinson specific code
  handles.Fit(:,ii) = scale * handles.Fit(:,ii);
  handles.other.xtra_info(2,ii) =  abs(scale*handles.Model.n_hf);
  waitbar(ii/n, h);
end
delete(h);
toc
disp(sprintf('Fitting of %i spectra is finished.', size(handles.yData, 2)));

guidata(hObject, handles);
Plot(handles);

% --------------------------------------------------------------------
function mGlobalFit_Callback(hObject, eventdata, handles)

if ~isfield(handles, 'Model')
  disp('Model is not set-up yet.'); return;
end

if ~isfield(handles, 'yData')
  disp('Data are not loaded yet.'); return;
end
pars.idx = find(handles.PG_float);

pars.N = length(handles.Model.Nucs);

tic
n = size(handles.yData, 2);
h = waitbar(0,'Please wait...','Name','Fitting is in progress');
for ii=1:n
  pars.x1 = handles.xData(:, ii)-mean(handles.xData(:, ii));
  pars.PG  = handles.PG(:, ii);
  pars.yy = handles.yData(:,ii);
% MATLAB minimization
%   x = handles.PG(pars.idx);
%   xx = fmincon(@err_full_fit,x,[],[],[],[],handles.PG_lb(pars.idx),handles.PG_ub(pars.idx),[],...
%     optimset('MaxFunEvals',1000,'Algorithm','active-set','Display','off', 'TolFun', 1e-6),...
%     pars);
%   xx_err = 0;
%   handles.PG(pars.idx, ii) = xx(2:end);
%   handles.PG_err(pars.idx, ii) = xx_err;
%   handles.Fit(:,ii) = full_fit(xx(2:end), pars)*4/pars.PG(4);
%   scale = x(1);
% end of MATLAB specific code  
% Robinson minimization  
  [xx, xx_err]=cw_grad_min(@full_fit, pars);
  handles.PG(pars.idx, ii) = xx;
  handles.PG_err(pars.idx, ii) = xx_err;
  pars.PG(pars.idx) = xx;
  handles.Fit(:,ii) = full_fit(xx, pars)*4/pars.PG(4);
  scale = (pars.yy'*handles.Fit(:,ii))/(handles.Fit(:,ii)'*handles.Fit(:,ii));
% end of Robinson specific code
  handles.Fit(:,ii) = scale * handles.Fit(:,ii);
  handles.other.xtra_info(2,ii) =  abs(scale*handles.Model.n_hf);
  waitbar(ii/n, h);
end
delete(h);
toc
disp(sprintf('Fitting of %i spectra is finished.', size(handles.yData, 2)));

guidata(hObject, handles);
Plot(handles);

% --------------------------------------------------------------------
function mFitCurrent_Callback(hObject, eventdata, handles)
if ~isfield(handles, 'Model')
  disp('Model is not set-up yet.'); return;
end

if ~isfield(handles, 'yData')
  disp('Data are not loaded yet.'); return;
end

if ~any(handles.PG_float)
  pars.idx = [1, 2, 7];
  handles.PG_float(:) = false;
  handles.PG_float(pars.idx) = true;
else
  pars.idx = find(handles.PG_float);
end

SliderValue = fix(get(handles.slSlice, 'Value'));

pars.N = length(handles.Model.Nucs);
pars.x1 = handles.xData(:, SliderValue)-mean(handles.xData(:, SliderValue));

tic
pars.PG  = handles.PG(:, SliderValue);
pars.yy = handles.yData(:,SliderValue);
% MATLAB minimization
% x = [1;0.001;0.02;0];
% rough_fit = full_fit(x(2:end), pars);
% x(1) = (pars.yy'*rough_fit)/(rough_fit'*rough_fit);
% PG_lb = [0; -0.2; 0; -pi/2];
% PG_ub = [x(1)*5; 0.2; 1; pi/2];
% xx = fminsearch(@err_full_fit,x,...
%   optimset('MaxFunEvals',1000,'Algorithm','active-set','Display','iter', 'TolFun', 1e-8),...
%   pars);
% % xx = fmincon(@err_full_fit,x,[],[],[],[], PG_lb,PG_ub,[],...
% %   optimset('MaxFunEvals',4000,'Algorithm','active-set','Display','off', 'TolFun', 1e-10),...
% %   pars);
% xx_err = 0;
% handles.PG(pars.idx, SliderValue) = xx(2:end);
% handles.PG_err(pars.idx, SliderValue) = xx_err;
% handles.Fit(:,SliderValue) = full_fit(xx(2:end), pars)*xx(1);
% end of MATLAB specific code
% Robinson minimization
[xx,xx_err]=cw_grad_min(@full_fit, pars);
handles.PG(pars.idx, SliderValue) = xx;
handles.PG_err(pars.idx, SliderValue) = xx_err;
pars.PG(pars.idx) = xx;
handles.Fit(:,SliderValue) = full_fit(xx, pars)*4/pars.PG(4);
scale = (pars.yy'*handles.Fit(:,SliderValue))/(handles.Fit(:,SliderValue)'*handles.Fit(:,SliderValue));
handles.Fit(:,SliderValue) = scale * handles.Fit(:,SliderValue);
% end of Robinson specific code
disp('Fitting is finished.');
toc

guidata(hObject, handles);
Plot(handles);

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% ------- L O A D   D A T A ------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

function mFileLoad_Callback(hObject, eventdata, handles)
[filename, pathname, filterindex] = uigetfile( ...
  {'*.s*', '250 MHz/X-band data (*.s*)'; ...
  '*.mat','Image file (*.mat)'; ...
  '*.mdl','Models (*.mdl)'; ...
  '*.d01','Rapid Scan (*.d01)'; ...
  '*.*',  'All Files (*.*)'}, ...
  'Pick a file', safeget(handles, 'PathName', 'c:\Users\Boris\Documents\MATLAB\'), ...
  'MultiSelect', 'on');
if ~(isequal(filename,0) || isequal(pathname,0))
  if ischar(filename), filename={filename}; end
  handles.PathName = pathname;
  handles.FilterIndex = filterindex;
  handles.xData = [];
  handles.yData = [];
  handles.Fit = [];
  other = safeget(handles, 'other', []);
  for ii=1:length(filename)
    [fpath, fname, fext]= fileparts(filename{ii});
    
    switch filterindex
      case 1,
        if ii==1,
          ff=inputdlg({'Mod factor:'},'Input', 1,{num2str(safeget(other,'modcal', 0.2311))});
          other.modcal    = str2double(ff{1});
        end
        idx = sscanf(fext, '.S%i');
        [other.pars_out,Z]=read_bin_spectrum(fullfile(pathname,fname),idx,idx,1) ;
        if isempty(handles.yData)
          handles.yData = Z(:,2);
          handles.xData = Z(:,1);
        else
          handles.yData(:,end+1) = Z(:,2);
          handles.xData(:,end+1) = Z(:,1);
        end
        handles.ModOmega = other.pars_out(17)*1E-3/2.802;% mod freq kHz
        handles.ModAmp   = other.pars_out(16)*str2double(ff{1});% mod amp gauss
        other.fname = fullfile(pathname,fname);
        handles.other = other;
      case 2,
        if ii==1
          ff=inputdlg({'Mod factor:','Threshold or -1 for external mask:'},'Input', 1,...
            {num2str(safeget(other,'modcal', 0.2311)),...
             num2str(safeget(other,'threshold', 0.15))});
          other.modcal    = str2double(ff{1});
          other.threshold = str2double(ff{2});
        end
        s1 = load(fullfile(pathname,fname));
        other.pars_out = safeget(s1, 'pars_out', zeros(25,1));
        dims = size(s1.mat_recFXD); ndim = ndims(s1.mat_recFXD);
        if 1
          N  = dims(ndim); NN = prod(dims(1:ndim-1));
          
          % mask
          mat_recFXD = permute(s1.mat_recFXD, [ndim-1:-1:1,4]);
          data = reshape(mat_recFXD, [NN, N]);
          if other.threshold > 0
            spec_amp  = max(data,[],2);
            im_mask   = spec_amp >= max(spec_amp) * other.threshold;
          else
            [FileName,PathName] = uigetfile({'*.mat', 'Matlab mask files (*.mat)'},'Load file', ...
              handles.PathName);
            
            if ~(isequal(FileName,0) || isequal(PathName,0))
              load_mask = load(fullfile(PathName, FileName), 'file_type');
              if isfield(load_mask, 'file_type')
                if strcmp(load_mask.file_type, 'ImageMask_v1.0')
                  load_mask = load(fullfile(PathName, FileName));
                end
              end
              im_mask = permute(load_mask.Mask, ndim-1:-1:1); im_mask = im_mask(:);
            else
              disp('Data are not loaded.');
              return;
            end
          end
          other.immask = max(s1.mat_recFXD,[],4);
          other.immask(permute(reshape(~im_mask, dims(1:ndim-1)), ndim-1:-1:1)) = 0;
          
          % y data
          dsdiff=diff(data(im_mask, :),1,2)';
          handles.yData=[dsdiff(1,:);dsdiff];
          
          % x data
          NN1 = size(handles.yData, 2);
          deltaH = s1.rec_info.rec.deltaH * sqrt(2);
          if deltaH == 0
            xData  =  1.024*sqrt(2)/N *(0:(N -1)); % typical field
            disp('Warning: field was not supplied.');
          else
            xData  =  deltaH/N *(0:(N -1));
          end
          handles.xData = repmat(xData' - mean(xData), 1, NN1);
          
          H_cntr = zeros(1, NN1);
        else
        end
%         [ ds, fn , H_cntr, other.pars_out, other.immask] = ...
%           cw_get_img(fullfile(pathname,fname),'mat_recFXD', other.threshold);
%         handles.yData = double(ds(:,2:2:end));
%         handles.xData = double(ds(:,1:2:end));
        handles.ModOmega = other.pars_out(17)*1E-3/2.802; % mod freq kHz
        handles.ModAmp   = other.pars_out(16)*str2double(ff{1}); % mod amp gauss
        handles.other = other;
        handles.other.xtra_info = zeros(8,size(handles.yData, 2));
        handles.other.xtra_info(1,:) =  H_cntr;
        handles.other.fname = fullfile(pathname,fname);
      case 4
        [ax, y, dsc] = kv_d01read(fullfile(pathname,[fname, '.exp']));
        Freq = kvgetvalue(safeget(dsc, 'SCAN_Frequency', '500 Hz'));
        ScanWidth = kvgetvalue(safeget(dsc, 'SCAN_ScanWidth', '500 Hz'));

        [x_RS, y_RS] = RapidScanDeconvolute(ax.x, y, Freq, ScanWidth);
        n = fix(length(x_RS)/2);
        if isempty(handles.yData)
          handles.yData = -real(diff(y_RS(1:n+1)));
          handles.xData = x_RS(1:n);
        else
          handles.yData(:,end+1) = -real(diff(y_RS(1:n+1)));
          handles.xData(:,end+1) = x_RS(1:n);
        end        
        handles.yData(:,end+1) = -real(diff(y_RS(end-n:end)));
        handles.xData(:,end+1) = x_RS(end-n+1:end);
        handles.ModOmega = 0.001; % mod freq kHz
        handles.ModAmp   = 0.001; % mod amp gauss
    end
    disp(['Image ', fullfile(pathname,fname), ' is loaded.']);
  end
  UpdateDataControls(handles);
  guidata(hObject, handles);
  GenerateParameters(handles);
  Plot(handles)
end

% --------------------------------------------------------------------
function [x_RS, y_RS] = RapidScanDeconvolute(x, y, RSfrequency, FieldSweep)

  sampling = mean(diff(x));
  gamma_const = 1.4211e-008;

  % deconvolute and average projections
  Npt  = size(y, 1);                % trace length
  Ntr  = size(y, 2);                % number of traces
  Nc = fix(1/RSfrequency/sampling); % points per cycle (up AND down)
  N = fix(Npt/Nc);                  % number of cycles in the trace
  
  Npt_process = 2*fix(N*Nc/2);      % number of points that will be processed
  
  d = gamma_const/(FieldSweep*RSfrequency);
  w = (1:Npt_process/2)*2*pi*RSfrequency/N;
  AF = exp(1i*d*w.^2); % analytical function
  A = [AF fliplr(conj(AF))]';
  
  f_domain = fft(y(1:Npt_process,:))./A(:,ones(Ntr,1));
  y_RS = ifft(f_domain(1:N:Npt_process, :)/N);
  nPt = length(y_RS);
  x_RS = [0:(nPt-1)]'/(nPt-1) * FieldSweep*2; 
%   x_RS = (0:Npt_process/N-1)*FieldSweep*2/(Npt_process-1);
%   x_RS = x_RS - mean(x_RS);

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% ---- F I T   F U N C T I O N S  ------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

function err = err_simple_fit(x, pars)

y = simple_fit(x(2:end), pars);
err = sqrt(sum((pars.yy - x(1)*y).^2));

% --------------------------------------------------------------------
function err = err_full_fit(x, pars)

y = full_fit(x(2:end), pars);
err = sqrt(sum((pars.yy - x(1)*y).^2));
% figure(2)
% plot(pars.x1, x(1)*y, pars.x1, pars.yy)
% pause(0.1)

% --------------------------------------------------------------------
function y = full_fit(x, pars)

PG = pars.PG;
PG(pars.idx) = x;

Spin_system = reshape(PG(9:end-1), [pars.N, 4])';
[y_pat, del_pat] = cw_shf(Spin_system);
x_pat = (pars.x1(1)-del_pat:del_pat:pars.x1(end)+del_pat)';

lw = cw_robinson(x_pat, PG(1), PG(2), PG(3), PG(4), 1);
y = conv2(imag(lw(:,1)*exp(-1i*PG(7))),y_pat,'same');
y = interp1(x_pat, y, pars.x1,'spline');

% --------------------------------------------------------------------
function y = simple_fit(x, pars)

PG = pars.PG;
PG(pars.idx) = x;

lw = cw_robinson(pars.x_pat, PG(1), PG(2), PG(3), PG(4), 1);
y = conv2(imag(lw(:,1)*exp(-1i*PG(7))),pars.y_pat,'same');
y = interp1(pars.x_pat, y, pars.x1,'spline');

% --------------------------------------------------------------------
function cpv = getcpv(pars)
[fpath, fname, fext] = fileparts(pars.fname);
[sfpath, sfname, sfext] = fileparts(pars.save_fname);
a = [fname, fext];                        
a = str2mat(a, 'mat_recFXD');                                 
a = str2mat(a, fpath);
a = str2mat(a, 'not specified');                              
a = str2mat(a, 'not specified');                              
a = str2mat(a, '0');                                          
a = str2mat(a, 'lor_fm');                                     
a = str2mat(a, 'not specified');                              
a = str2mat(a, num2str(pars.modcal));                                      
a = str2mat(a, '25.1189');                                    
a = str2mat(a, 'OX063H');                                     
a = str2mat(a, '6206');                                       
a = str2mat(a, '8');                                          
a = str2mat(a, '8');                                          
a = str2mat(a, '1');                                          
a = str2mat(a, '4');                                          
a = str2mat(a, [sfname, sfext]);      
a = str2mat(a, '0.5');                                        
a = str2mat(a, '-2.5');                                       
a = str2mat(a, '3');                                          
a = str2mat(a, 'path\filename');                              
a = str2mat(a, '0.01');                                       
a = str2mat(a, 'UCLGR');                                      
a = str2mat(a, '0.02');                                       
a = str2mat(a, '5.12');                                       
a = str2mat(a, 'user file name');                             
a = str2mat(a, 'not specified');                              
a = str2mat(a, 'image_epr');                                  
a = str2mat(a, '0');                                          
a = str2mat(a, 'k-r-k-');                                     
a = str2mat(a, '0.050');                                      
a = str2mat(a, '1e-006');                                     
a = str2mat(a, 'not specified');                              
cpv = str2mat(a, 'null'); 

function cpv_text = getcpv_text()
a = 'data mat file name';       
a = str2mat(a, 'data array name');          
a = str2mat(a, 'directory');                
a = str2mat(a, 'end #');                    
a = str2mat(a, 'fn_hi');                    
a = str2mat(a, 'gaussian sigma(G)');        
a = str2mat(a, 'lor_fun');                  
a = str2mat(a, 'mat file');                 
a = str2mat(a, 'modulation cal factor');    
a = str2mat(a, 'MP(mW)');                   
a = str2mat(a, 'nitroxide');                
a = str2mat(a, 'n_ds');                     
a = str2mat(a, 'n_info');                   
a = str2mat(a, 'n_parts');                  
a = str2mat(a, 'n_peaks');                  
a = str2mat(a, 'n_shf_atoms');              
a = str2mat(a, 'parameter file name');      
a = str2mat(a, 'pass_line');                
a = str2mat(a, 'phi(A/D)');                 
a = str2mat(a, 'phi(Zmn');                 
a = str2mat(a, 'prefix');               
a = str2mat(a, 'R1(G)');                
a = str2mat(a, 'resonator');            
a = str2mat(a, 'RMA(G)');               
a = str2mat(a, 'RMF(kHz)');             
a = str2mat(a, 'root_fn');              
a = str2mat(a, 'root_mat_fn');          
a = str2mat(a, 'Source');               
a = str2mat(a, 'start #');              
a = str2mat(a, 'style');                
a = str2mat(a, 'threshold');            
a = str2mat(a, 'tolerance');            
a = str2mat(a, 'Xepr file name');       
cpv_text = str2mat(a, 'Zmn_phase');  

function dsc = get_description(idx)
dsc.cc = 1;
dsc.Title = 'unknown';

switch idx
  case 1
    dsc.Title = 'Zero crossover [mG]';
    dsc.cc = 1E3;
  case 2
    dsc.Title = 'Lorentzian linewidth [mG]';
    dsc.cc = 1E3;
  case 7
    dsc.Title = 'Phase [degree]';
    dsc.cc = 180/pi;
end

% --------------------------------------------------------------------
function mShowStat_Callback(hObject, eventdata, handles)

f_pars = find(handles.PG_float);

if ~isempty(f_pars)
  fit_set = num2str(f_pars(1)); for ii=2:length(f_pars); fit_set = [fit_set, ', ', num2str(f_pars(ii))]; end
  for ii=1:length(f_pars)
    pars = get_description(f_pars(ii));
    data = handles.PG(f_pars(ii), :)*pars.cc;
    figure; hist(data, min(100, fix(length(data)*0.6))); xlabel(pars.Title);
    text(0.75, 0.9, sprintf('median = %6.2f', median(data)), 'Units', 'normalized');
    text(0.75, 0.84, sprintf('mean   = %6.2f', mean(data)), 'Units', 'normalized');
    text(0.75, 0.78, sprintf('std   = %6.2f', std(data)), 'Units', 'normalized');
    title([epr_ShortFileName(handles.other.fname, 40), ' [',fit_set,']'], 'Interpreter', 'none')
  end
end



