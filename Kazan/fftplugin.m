function varargout = fftplugin(varargin)
% POWDERBOX Application M-file for powderbox.fig
%    Intended to use only with 'kazan' viewer

% KAZAN dataviewer with plugins By Boris Epel & Alexey Silakov
% MPI of Bioinorganic Chemistry, Muelhaim an der Ruhr, 2003
% Free for non-commercial use. Use the program at your own risk. 
% The authors retains all rights. 
% Contact: epel@mpi-muelheim.mpg.de

% Last Modified by GUIDE v2.0 21-Dec-2004 13:28:22
% Boris Epel 5-dec-2003, MPI
% boep 21-dec-2003, MPI

if nargin == 0  % LAUNCH GUI
    token = 'FFTPLUGIN_open';
    oldfig = getappdata(0, token);
    oldfig = oldfig(ishandle(oldfig));
    
    [fpath] = fileparts(which('kazan'));
    inifilename = [fpath '\kazan.ini'];
    ini = inimanage(inifilename);
    opc = 2;
    try opc = ini.KazanViewer.MultyKazan; catch end
    if (opc == 1 || opc == 2) && ~isempty(oldfig), set(oldfig(1),'Visible','on'); varargout{1}=oldfig(1); return; end

	fig = openfig(mfilename,'new');
    setappdata(0, token, [oldfig, fig]);
  
  % Generate a structure of handles to pass to callbacks, and store it. 
  handles = guihandles(fig);
  handles.hh = 0;
  handles.ploter = 0;
  handles.num = 1;
  handles.div = 0;
  handles.Spec = {};
  handles.sim{1}.Script = get(handles.list, 'String');
  handles.sim{1}.x = [];
  handles.sim{1}.x1 = [];
  handles.sim{1}.y = [];
  handles.sim{1}.stype = 'sim';
  handles.Color =   [1, 0, 0;...
      1, 1, 0;...
      0, 1, 0;...
      0, 0, 0;...
      1, 0, 1;...
      0, 1, 1;...
      1, 1, 0];
  handles.SumCol = [1, 0, 0];
  handles.script = {};

    %**********************************************************************
    % Main Menu
    %**********************************************************************
    % menu File
    handles.mFile = uimenu(handles.figure1, 'Label','File', 'Tag', 'mFile');
    handles.mFileAddBaseline = uimenu(handles.mFile, 'Label','Add baseline', 'Tag', 'mFileAddBaseline');
    handles.mFileBaseline = uimenu(handles.mFileAddBaseline, 'Label','From File', 'Tag', 'mFileAddBaseline', ...
        'Callback', 'fftplugin(''AddSim_Callback'',gcbo,[],guidata(gcbo))');
    handles.mBl1DPolyfit = uimenu(handles.mFileAddBaseline, 'Label','1D polyfit', 'Tag', 'mBl1DPolyfit', ...
        'Callback', 'fftplugin(''AddIntBl_Callback'',gcbo,[],guidata(gcbo))');
    handles.mBl2DPolyfit = uimenu(handles.mFileAddBaseline, 'Label','2D polyfit', 'Tag', 'mBl2DPolyfit', ...
        'Callback', 'fftplugin(''AddIntBl_Callback'',gcbo,[],guidata(gcbo))');
    handles.mFileDelete = uimenu(handles.mFile, 'Label','Delete', 'Tag', 'mFileDelete', ...
        'Callback', 'fftplugin(''butDel_Callback'',gcbo,[],guidata(gcbo))');
    handles.LoSc = uimenu(handles.mFile, 'Label','Load', 'Tag', 'LoSc', ...
        'Callback', 'fftplugin(''LoSc_Callback'',gcbo,[],guidata(gcbo))', ...
        'Accelerator', 'L');
    handles.SvSc = uimenu(handles.mFile, 'Label','Save', 'Tag', 'SvSc', ...
        'Callback', 'fftplugin(''SvSc_Callback'',gcbo,[],guidata(gcbo))', ...
        'Accelerator', 'S');
    handles.mFileCopy = uimenu(handles.mFile, 'Label','Copy', 'Tag', 'mFileCopy', ...
        'Callback', 'fftplugin(''mFileCopy_Callback'',gcbo,[],guidata(gcbo))');
    handles.mFileScript = uimenu(handles.mFile, 'Label','Show script', 'Tag', 'mFileScript', ...
        'Callback', 'fftplugin(''mFileScript_Callback'',gcbo,[],guidata(gcbo))', ...
        'Separator', 'on');
    handles.mFileExecScript = uimenu(handles.mFile, 'Label','Execute script', 'Tag', 'mFileExecScript', ...
        'Callback', 'fftplugin(''mFileExecScript_Callback'',gcbo,[],guidata(gcbo))');
    handles.mOptions = uimenu(handles.figure1, 'Label','Options', 'Tag', 'mOptions');
    handles.mSimSubBl = uimenu(handles.mOptions, 'Label','Subtract baseline', 'Tag', 'mSimSubBl', ...
        'Callback', 'fftplugin(''mFileExecScript_Callback'',gcbo,[],guidata(gcbo))');
    handles.mCalc = uimenu(handles.mOptions, 'Label','Calculate', 'Tag', 'mCalc', ...
        'Callback', 'fftplugin(''check_Callback'',gcbo,[],guidata(gcbo))', ...
        'Accelerator', 'C', 'Separator', 'on');
    handles.mAbout = uimenu(handles.figure1, 'Label','About', 'Tag', 'mAbout', ...
        'Callback', 'fftplugin(''mAbout_Callback'',gcbo,[],guidata(gcbo))');
        
  set(handles.butCol, 'BackgroundColor', handles.Color(1, :));
  guidata(fig, handles);
  list_Callback([], [], handles);
  
  if nargout > 0
    varargout{1} = fig;
  end
  
elseif ischar(varargin{1}) % INVOKE NAMED SUBFUNCTION OR CALLBACK
  
  try
%     [varargout{1:nargout}] = 
    feval(varargin{:}); % FEVAL switchyard
  catch
    disp(lasterr);
  end
  
end


%| ABOUT CALLBACKS:
%| GUIDE automatically appends subfunction prototypes to this file, and 
%| sets objects' callback properties to call them through the FEVAL 
%| switchyard above. This comment describes that mechanism.
%|
%| Each callback subfunction declaration has the following form:
%| <SUBFUNCTION_NAME>(H, EVENTDATA, HANDLES, VARARGIN)
%|
%| The subfunction name is composed using the object's Tag and the 
%| callback type separated by '_', e.g. 'slider2_Callback',
%| 'figure1_CloseRequestFcn', 'axis1_ButtondownFcn'.
%|
%| H is the callback object's handle (obtained using GCBO).
%|
%| EVENTDATA is empty, but reserved for future use.
%|
%| HANDLES is a structure containing handles of components in GUI using
%| tags as fieldnames, e.g. handles.figure1, handles.slider2. This
%| structure is created at GUI startup using GUIHANDLES and stored in
%| the figure's application data using GUIDATA. A copy of the structure
%| is passed to each callback.  You can store additional information in
%| this structure at GUI startup, and you can change the structure
%| during callbacks.  Call guidata(h, handles) after changing your
%| copy to replace the stored original so that subsequent callbacks see
%| the updates. Type "help guihandles" and "help guidata" for more
%| information.
%|
%| VARARGIN contains any extra arguments you have passed to the
%| callback. Specify the extra arguments by editing the callback
%| property in the inspector. By default, GUIDE sets the property to:
%| <MFILENAME>('<SUBFUNCTION_NAME>', gcbo, [], guidata(gcbo))
%| Add any extra arguments after the last argument, before the final
%| closing parenthesis.
% --------------------------------------------------------------------
function FileUpdate(varargin)

% --------------------------------------------------------------------
function SelectionUpdate(hObject, par, handles)

% --------------------------------------------------------------------
function varargout = list_Callback(h, eventdata, handles, varargin)
Res = get(handles.list, 'String');
val = get(handles.list, 'Value');
if size(Res)<2, 
   val = 1; 
end

[name, str1] = strtok(Res{val}, '=');
str2 = [strtrim(str1(2:end)), ' '];

if strcmp(strtrim(name), 'f')
  set(handles.ePars, 'String', str2);
  set(handles.ePars, 'UserData', 'function');
elseif str2(1)==''''
  primes = findstr(str2, '''');
  if size(primes, 2) < 2, primes(2) = size(str2, 2); end
  set(handles.ePars, 'String', str2(2:(primes(2)-1)));
  set(handles.ePars, 'UserData', 'string');
else
  dig = str2num(str2);
  set(handles.ePars, 'String', num2str(dig, 6));
  set(handles.ePars, 'UserData', 'value');
end

% --------------------------------------------------------------------
function varargout = ePars_Callback(h, eventdata, handles, varargin)
% Stub for Callback of the uicontrol handles.ePars.
Res = get(handles.list, 'String');
val = get(handles.list, 'Value');
newval = strtrim(get(handles.ePars, 'String'));

[name] = strtok(Res{val}, '=');
name = strtrim(name);

switch get(handles.ePars, 'UserData')
  case 'function'
    set(handles.ePars, 'String', newval);
    tokens = fsplitter([name ' = ' newval]);
    % load script 
    for k=1:size(Res,1),
      if ~strcmp(strtrim(strtok(Res{k},'=')),'f')
        try eval([Res{k} ';']); catch disp(['Error ! ', Res{k} ';']); end
      end
    end
    Res = {[name ' = ' newval]};
    for k = 1: size(tokens, 2)
      try var = eval(tokens{k}); catch disp(['Error ! ',tokens{k}]); var = 0; end
      Res{end+1} = [tokens{k},' = ', num2str(var)];
    end
  case 'string'
    set(handles.ePars, 'string', newval);
    Res{val} = [name ' = ''', newval,''''];
  otherwise
    nums = str2num(newval);
    set(handles.ePars, 'String', num2str(nums));
    Res{val} = [name ' = ' num2str(nums)];
end
set(handles.list, 'String', Res);
list_Callback(h, eventdata, handles);
handles.sim{get(handles.popSel, 'Value')}.Script = Res;
guidata(handles.figure1, handles);


if strcmp(get(handles.mCalc, 'Checked'), 'on'), 
   stype = get(handles.popSel, 'Value');
   switch handles.sim{stype}.stype
      case 'sim'
         plothis(handles);
      case 'baseline'
         handles = bscalc(get(handles.popSel, 'Value'), handles);
         plothis(handles);
   end
end

% --------------------------------------------------------------------
function varargout = slPars_Callback(h, eventdata, handles, varargin)
 
if ~strcmp(get(handles.ePars, 'UserData'),'value'), return; end;
shift = get(handles.slPars, 'Value');
set(handles.slPars, 'Value',0);
num = get(handles.popup, 'Value');
str = get(handles.popup, 'String');
dim = str2num([str{num}]);

CurVal = str2double(get(handles.ePars, 'String'));
Val = CurVal + dim*shift;

set(handles.ePars, 'String', num2str(Val));
ePars_Callback(h, eventdata, handles, varargin);

% --------------------------------------------------------------------
function handles = bscalc(num, handles)

hand = guidata(handles.hh);
x = hand.src.ax.x;
x1 = hand.src.ax.y;
y = hand.src.y;
func = ';';
% load script 
str = get(handles.list, 'String');
for k=1:size(str,1),
  if strcmp(strtrim(strtok(str{k},'=')),'f')
    func = str{k};
  else
    try eval([str{k} ';']); catch disp(['Error ! ', str{k}, ';']);end
  end
end
try eval([func ';']); catch disp(['Error ! ', func ';']); end

handles.sim{num}.x = x;
handles.sim{num}.x1 = x1;
handles.sim{num}.amp = 1; %abs(max(f)-min(f));
handles.sim{num}.y = f;
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function plothis(handles)
handles = guidata(handles.figure1);
colors = handles.Color;

hand = guidata(handles.hh);
ax = hand.src.ax;
y = hand.src.y;

% baseline correction
basln = strcmp(get(handles.mSimSubBl, 'Checked'), 'on');

% load script 
fft = [];
str = get(handles.list, 'String');

for k=1:size(str,1)
    try eval([str{k} ';']); catch , end
end

bline = zeros(size(y));
for i = 2:handles.num
    if strcmp(handles.sim{i}.stype, 'baseline')
        bline = bline + handles.sim{i}.y;
    end
end

switch safeget(fft,'bsline', 'none')
  case 'none'
  case '0'
    zeroO = mean(y, 1);
    bline = repmat(zeroO, [size(y,1), 1]);
  case 'polyfit1D'
    %         bline = polyfit1(x, y, n, dir) + bline;
  case 'polyfit2D'
    n = safeget(fft, 'n_pol', 0);
    bline = polyfit2(ax.x, ax.y, y, n) + bline;
  case 'last10'
    np = floor(length(ax.x)*0.15);
    bline = mean(y(np:end-1,:));
  case 'last15'
    np = floor(length(ax.x)*0.15);
    bline = mean(y(np:end-1,:));
end

y = y - bline;

hand.out = {};

if get(handles.butFFT, 'Value')
    %%%%%%%%%%%%%%%% FFT %%%%%%%%%%%%%%%%%%%%%%%
    handles = guidata(handles.figure1);
    
    fft.fft = 1;
    opt = safeget(fft, 'opt', 'real');
    if strcmp(opt, 'reim2') % 2D FFT: fft(fft(real(y)))
        fft2 = 1;
        fft.opt = 'real';
        [tax, ty] = kv_fft(ax, y, fft);
        fft.opt = 'imag';
        tax.y = tax.x;
        tax.x = ax.y;
        [rax, ry] = kv_fft(tax, ty.', fft);
    else
        fft2 = 0;
        if hand.Projection == 1
            [rax, ry] = kv_fft(ax, y, fft);
        else
            ax1 = ax;
            ax1.x      = ax.y; ax1.xlabel = safeget(ax, 'ylabel', 'x');
            ax1.y      = ax.x; ax1.ylabel = safeget(ax, 'xlabel', 'y');
            [ax1, ry] = kv_fft(ax1, y', fft);
            ry = ry'; 
            rax = ax;
            rax.x = ax1.y; rax.xlabel = safeget(ax1, 'ylabel', 'x');
            rax.y = ax1.x; rax.ylabel = safeget(ax1, 'xlabel', 'y');
        end
    end
    hand.out{1} = hand.src;
    hand.out{1}.ax = rax;
    hand.out{1}.ax.dx = 0;
    hand.out{1}.y = ry;
    hand.out{1}.ax.Color = [0, 0, 1]; % always blue
    hand.out{1}.ax.s = 0;
    hand.out{1}.title = 'FFT';
else
    hand.out{1}.ax = hand.src.ax;
    hand.out{1}.y = y; 
    hand.out{1}.ax.Color = [0, 0, 1]; % always blue
    hand.out{1}.ax.s = 0;
    hand.out{1}.title = 'Src';
    
    handles = guidata(handles.figure1);
    
    % load script 
    str = get(handles.list, 'String');
    fft = [];
    for k=1:size(str,1),
        try eval([str{k} ';']); catch disp(['Error ! ',str{k},';']); end
    end
    
    opt = safeget(fft, 'opt', 'real');
    fft.fft = 0; % do no FFT
    if strcmp(opt, 'reim2'), % if 2D FFT
        fft.opt = 'real';
        [tax, ty, tawin] = kv_fft(ax, y, fft);
        fft.opt = 'imag';
        tax.y = tax.x;
        tax.x = ax.y;
        [rax, rty, ttawin] = kv_fft(tax, ty.', fft);
        ry = rty.';
        rawin = ttawin.*tawin;
        % now data is rotated, not good
    else
        if hand.Projection == 1
            [rax, ry, rawin] = kv_fft(ax, y, fft);
        else
            ax1 = ax;
            ax1.x      = ax.y; ax1.xlabel = safeget(ax, 'ylabel', 'x');
            ax1.y      = ax.x; ax1.ylabel = safeget(ax, 'xlabel', 'y');
            [ax1, ry, rawin] = kv_fft(ax1, y', fft);
            ry = ry'; rawin = rawin';
            rax = ax;
            rax.x      = ax1.y; rax.xlabel = safeget(ax1, 'ylabel', 'x');
        end
    end
    nawin = max(rawin);    % try to cause error
    nyr = max(max(real(hand.out{1}.y)));
    nyi = max(max(imag(hand.out{1}.y)));
    hand.out{end+1}.ax = rax;
    hand.out{end}.y = rawin*(nyr+j*nyi);
    hand.out{end}.ax.Color = [0, 1, 0]; % always green
    hand.out{end}.ax.s = 0;
    hand.out{end}.title = 'Awin';
    % application of window
    hand.out{end+1}.ax = rax;
    hand.out{end}.y = ry;
    hand.out{end}.ax.Color = [0, 0, 0]; % always black
    hand.out{end}.ax.s = 0;
    hand.out{end}.title = 'Out';
    % show baseline 
    if ~basln,
        hand.out{end+1}.ax = hand.src.ax;
        hand.out{end}.ax.y = hand.src.ax.y;    
        hand.out{end}.y = bline;
        hand.out{end}.ax.Color = [1, 0, 0]; % red
        hand.out{end}.ax.s = 0;
        hand.out{end}.title = 'Bline';
    end
end


guidata(handles.hh, hand);
[fpath,name] = fileparts(get(handles.hh, 'FileName'));
eval([name '(''SetDataSource'', 2, hand)']);

% --------------------------------------------------------------------
function LoSc_Callback(hObject, eventdata, handles)
full = get(handles.figure1, 'FileName');
[fpath] = fileparts(full); 
[fname,pname] = uigetfile([fpath '\pepper_scr\*.m'],'Open Script');
if fname == 0, return; end

if ischar(get(handles.popSel, 'String')), 
  str = {get(handles.popSel, 'String')};
else
  str = get(handles.popSel, 'String');
end
val = get(handles.popSel, 'Value');

[name, scr] = LoadScript([pname '\' fname], handles.sim{val}.stype);
set(handles.list, 'String', scr, 'Value', 1);
list_Callback(hObject, eventdata, handles);

str{val} = [num2str(val), ': ', fname];
set(handles.popSel, 'String', str);
handles.sim{val}.Script = scr;
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function varargout = LoadScript(fullname, stype)
fid = fopen(fullname, 'r');
[fpath,fname] = fileparts(fullname); 
if fid == -1, disp('File not found'); return; end
str = {};
while feof(fid) < 1
  st = fgetl(fid);
  if ~isempty(st) & st(1) ~= '%'
    str{end+1} = st;
  end
end
fclose(fid);

if nargout > 0 
  varargout = {fname , str.'};
end

% --------------------------------------------------------------------
function SvSc_Callback(hObject, eventdata, handles)

[fname,pname] = uiputfile(['\*.m'],'Save Script');
if fname == 0, return; end
[fpath,fname,ext] = fileparts([pname, fname]); 
if strcmp(ext ,'') , ext = '.m'; end
fid = fopen([pname, fname, ext], 'w');
str = get(handles.list, 'String');
for i = 1:size(str, 1)
  fprintf(fid, '%s\r\n', str{i});
end
fclose(fid);

% ------------------------------------------------------------
function check_Callback(hObject, eventdata, handles)
c = {'off', 'on'};
col = {[1 0.7 0.7], [0.7, 1, 0.7]};
stat = strcmp(get(handles.mCalc, 'Checked'), 'on');
set(handles.mCalc, 'Checked', [c{~stat + 1}]);
set(handles.butCalc, 'Value', ~stat,'BackgroundColor', [col{~stat + 1}]);

if ~stat,
    ePars_Callback(hObject, eventdata, handles)
else
    set(handles.butFFT, 'Value',0);
end

% ------------------------------------------------------------
function figure1_CloseRequestFcn(hObject, eventdata, handles)
try
  hand = guidata(handles.hh);
  hand.rx = {};
  hand.ry = {};
  hand.rc = {};
  hand.rs = {};
  guidata(handles.hh, hand);
  eval([name '(''SetDataSource'', 2, hand)']);
end
delete(handles.figure1);

% ------------------------------------------------------------
function butCol_Callback(hObject, eventdata, handles)
num = get(handles.popSel, 'Value');
c = uisetcolor(handles.Color(num, :));
if size(c, 2) > 1
  
  handles.Color(num, :) = c;
  set(handles.butCol, 'BackgroundColor', c, 'ForegroundColor', [1 1 1] - c);
  guidata(handles.figure1, handles);
  plothis(handles);
end

% --- Executes on button press in butDel.
function butDel_Callback(hObject, eventdata, handles)
num = handles.num;
val = get(handles.popSel, 'Value');
if num == 1, return; end

switch val
  case 1,
    ss = 2:num;
  case num
    ss = 1:num-1;
  otherwise
    ss = [1:val-1, val+1:num];
end

handles.sim = {handles.sim{ss}};
handles.num = handles.num - 1;
str = get(handles.popSel, 'String');
set(handles.popSel, 'String', str{ss}, 'Value', 1);
popSel_Callback(hObject, eventdata, handles);
guidata(handles.figure1, handles);

% ------------------------------------------------------------
function popSel_Callback(hObject, eventdata, handles)
num = get(handles.popSel, 'Value');
set(handles.list, 'String', handles.sim{num}.Script, 'Value', min(2, size(handles.sim{num}.Script, 1)));
list_Callback(hObject, eventdata, handles);
set(handles.butCol, 'BackgroundColor', handles.Color(num, :)); 

% --------------------------------------------------------------------
function AddSim_Callback(hObject, eventdata, handles)
full = get(handles.figure1, 'FileName');
[fpath] = fileparts(full); 
[fname, pname] = uigetfile([fpath '\pepper_scr\*.m'],'Open Script');
if fname == 0, return; end

switch hObject
case handles.mFileBaseline
  stype = 'baseline';
otherwise
  stype = 'sim';
end

[name, script] = LoadScript([pname fname], stype);
set(handles.list, 'String', script, 'Value', 1);
list_Callback(hObject, eventdata, handles);

if ischar(get(handles.popSel, 'String')),
  str = {get(handles.popSel, 'String')};
else
  str = get(handles.popSel, 'String');
end

str{handles.num + 1} = [num2str(handles.num + 1), ': ', name];
set(handles.popSel, 'String', str, 'Value', handles.num + 1);

hand = guidata(handles.hh);
handles.sim{handles.num + 1}.Script = script;
handles.sim{handles.num + 1}.x = hand.src.ax.x;
handles.sim{handles.num + 1}.y = zeros(size(hand.src.ax.x));
handles.sim{handles.num + 1}.stype = stype;
handles.num = handles.num + 1;
guidata(handles.figure1, handles);
popSel_Callback(hObject, eventdata, handles);

% --------------------------------------------------------------------
function mSimShow_Callback(hObject, eventdata, handles)
c = {'off', 'on'};
stat = strcmp(get(hObject, 'Checked'), 'on');
set(hObject, 'Checked', [c{~stat + 1}]);
if ~stat 
  plothis(handles);
end

% --------------------------------------------------------------------
function varargout = mFileSalt_Pepper_Callback(h, eventdata, handles, varargin)
func = {'Pepper', 'Salt'}; % For future, when Num. of Functions will be > 2
Name = get(h, 'Label');
for i = 1:size(func, 2)
  if strcmp(func{i}, Name), 
    set(h, 'Checked', 'on');
    handles.CalcType = lower(Name);
  else
    eval(['set(handles.',['mFile', func{i}], ', ''Checked'', ''off'');']);
  end
end
guidata(h, handles);

% --------------------------------------------------------------------
function varargout = mFileCopy_Callback(h, eventdata, handles, varargin)


% --------------------------------------------------------------------
function mColSum_Callback(hObject, eventdata, handles)
num = get(handles.popSel, 'Value');
c = uisetcolor(handles.Color(num, :));
if size(c, 2) > 1
  handles.SumCol = c;
  %set(handles.butCol, 'BackgroundColor', c, 'ForegroundColor', [1 1 1] - c);
  guidata(handles.figure1, handles);
  plothis(handles);
end

% --------------------------------------------------------------------
function mFixAxis_Callback(h, eventdata, handles)
c = {'off', 'on'};
stat = strcmp(get(handles.mFixAxis, 'Checked'), 'on');
set(handles.mFixAxis, 'Checked', [c{~stat + 1}]);
if ~stat 
  plothis(handles);
end

function varargout = nearme(X, xi)
% nearme(X, xi)
% k = nearme(X, xi)
% [k, v] = nearme(X, xi)
% 
%   return value (v) and index (k) 
%   of nearest to xi number of array X 

[c, k] = min(abs(X - ones(size(X))*xi));
v = X(k);
switch nargout 
  case 0
    varargout{1} = k;
  case 1
    varargout{1} = k;
  case 2
    varargout{1} = k;
    varargout{2} = v; 
end

% --- Executes on button press in butCalc.
function butCalc_Callback(hObject, eventdata, handles)
check_Callback(hObject, eventdata, handles);

% --------------------------------------------------------------------
function butFFT_Callback(hObject, eventdata, handles)
c = {'off', 'on'};
stat = strcmp(get(handles.mCalc, 'Checked'), 'on');

if ~stat & get(handles.butFFT, 'Value')
  col = {[1 0.7 0.7], [0.7, 1, 0.7]};
  set(handles.mCalc, 'Checked', [c{~stat + 1}]);
  set(handles.butCalc, 'Value', ~stat,'BackgroundColor', [col{~stat + 1}]);
end

ePars_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function mFileScript_Callback(hObject, eventdata, handles)
if hObject==handles.mFileExecScript
    for k=1:size(handles.script,2),
        disp(handles.script{k})
        evalin('base',handles.script{k})
    end
else
    disp(' ')
    for k=1:size(handles.script,2),
        disp(handles.script{k})
    end
end

% --------------------------------------------------------------------
function mAbout_Callback(hObject, eventdata, handles)
msgbox({'FFTPLUGIN'; 'by Boris Epel and Alexey Silakov, 2003-04';''; ...
        'available parameters are:'; ...
        'awin - apodization window';...
        '    = bla/bar/con/cos/ham/han/wel/exp/gau/kai'; ...
        'awidth, aalpha, ashift - apodization window pars';...
        'zerofill - fill up to 2^(log2(size)+zerofill) points <0>';...
        'rshift, lshift - shift of data (points) <0>';...
        'phase0 - zero-order phase correction (deg) <0>';...
        'xscale - x axis coefficient <1>';...
        'opt = real/imag/qseq/cta/reim2 - type of FT <real>';...
        '   qseq - sequential acquisition';...
        '   cta  - cross-term averaging';...
        'cta - cross-term averaging parameter <datasize/4>';...
        '   reim2 - 2D fft. FFT(FFT(real(y)))';...   
        'bsline = none/polyfit2D - automatic baseline';...
        '   n_pol (for ''polyfit2D'') - degree of the polynom to fit';...
        ''; ...
        'In plugin'},...
      'About', 'help')

% --------------------------------------------------------------------
function butFFTall_Callback(hObject, eventdata, handles)
ePars_Callback(hObject, eventdata, handles)

%%%%%%% AlSi 10.09.2004 %%%%%%%%%%%%%%%
function out = polyfit1(x, y, n, dir)
switch dir
   case 1
      str = '(:, ii)';
      tr = '';
      dd = 2;
   otherwise
      str = '(ii, :)';
      tr = '.''';
      dd = 1;
end
for ii = 1:size(y, dd)
   eval(['ty(:, 1) = real(y', str, tr, ');']);
   pp = polyfit(x, ty, n);
   eval(['out', str,' = polyval(pp, x',tr,');']);
end

%-----------------------------------------------------
function out = polyfit2(x, x1, y, n)
tout = y*0;
if size(y, 2) > 1
   for ii = 1:size(y, 2)
      pp = polyfit(x, real(y(:, ii)), n);
      tout(:, ii) = polyval(pp, x);
   end   
end
out = tout;
if size(y, 1) >1
for jj = 1:size(y, 1)
   pp = polyfit(x1, (real(y(jj, :)) -tout(jj, :)).', n);
   out(jj, :) = polyval(pp, x1).' + tout(jj, :);
end
end

%-----------------------------------------------------
function AddIntBl_Callback(hObject, eventdata, handles)
switch hObject
    case handles.mBl1DPolyfit
        Scr = {'f = polyfit1(x, y, n, 1)'; 'n = 2'};
        titlestr = '1D polyfit';
    case handles.mBl2DPolyfit
        Scr = {'f = polyfit2(x, x1, y, n)'; 'n = 2'};        
        titlestr = '2D polyfit';
    otherwise
end
if ischar(get(handles.popSel, 'String')) 
  str = {get(handles.popSel, 'String')};
else
  str = get(handles.popSel, 'String');
end
val = length(str) + 1;
%val = get(handles.popSel, 'Value');

set(handles.list, 'String', Scr, 'Value', 1);
list_Callback(hObject, eventdata, handles);

str{val} = [num2str(val), ': ', titlestr];
set(handles.popSel, 'String', str, 'Value', val);


hand = guidata(handles.hh);
handles.sim{val}.Script = Scr;
handles.sim{val}.x = hand.src.ax.x;
handles.sim{val}.x1 = hand.src.ax.y;
handles.sim{val}.y = (hand.src.y)*0;
handles.sim{val}.stype = 'baseline';
handles.num = handles.num + 1;
guidata(handles.figure1, handles);
popSel_Callback(hObject, eventdata, handles);

guidata(handles.figure1, handles);
% --------------------------------------------------------------------

function [prj, sel, trans] = getslider_kazan(hand)
str = {'X', 'Y'};
trans = '';
prj = str{hand.Projection};
if hand.PlotType ==1
    switch hand.Projection
        case 1
            sel = ['(:, ','min(',  num2str(hand.Selection),', end))'];
        case 2
            sel = ['(min(', num2str(hand.Selection),', end), :)'];
            trans = '''';
    end
else
    sel = '(:, :)';
end
       
