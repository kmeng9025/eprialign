function varargout = cwplugin(varargin)
% CWPLUGIN Application M-file for cwplugin.fig
%    FIG = CWPLUGIN launch cwplugin GUI.
%    CWPLUGIN('callback_name', ...) invoke the named callback.

% Last Modified by GUIDE v2.5 19-Feb-2023 10:38:55

if nargin == 0  % LAUNCH GUI
  token = 'CWPLUGIN_open';
  oldfig = getappdata(0, token);
  oldfig = oldfig(ishandle(oldfig));
  
  [fpath] = fileparts(which('kazan'));
  inifilename = [fpath '\kazan.ini'];
  ini = inimanage(inifilename);
  opc = 2;
  try opc = ini.KazanViewer.MultyKazan; catch; end
  if (opc == 1 || opc == 2) && ~isempty(oldfig), set(oldfig(1),'Visible','on'); varargout{1}=oldfig(1); return; end
  
  fig = openfig(mfilename,'new');
  setappdata(0, token, [oldfig, fig]);
  
  % Use system color scheme for figure:
  set(fig,'Color',get(0,'defaultUicontrolBackgroundColor'));
  
  % Generate a structure of handles to pass to callbacks, and store it.
  handles = guihandles(fig);
  
  handles.DefaultFont = 8;
  handles.FitMethod = '';
  handles.src.type ='absorption';
  handles.src.source ='source';
  handles.src.filter = '';
  handles.dest.type ='absorption';
  handles.fit.x = [];
  handles.fit.y = [];
  for kk=1:6
    handles.range(kk).pnts = [];
    handles.range(kk).bracket = [];
  end
  handles.Script = get(handles.lbPars, 'String');
  handles.Pages{1} = {'f=c'};
  handles.Pages{2} = handles.Script;
  handles.Pages{3} = {'f=c*x.^2+b*x+a'};
  handles.Pages{4} = {'f=b*lshape(x,x0,fw,0,1)+a'};
  handles.Pages{5} = {'f=b*lshape(x,x0,fw,1,1)+a'};
  handles.Pages{6} = {'f=c'};
  handles.ActivePage = 2;
  handles.AreaFit = [];
  set(handles.pb1,'Value',1)
  handles.Color =   [1, 0, 0;...
    1, 1, 0;...
    0, 1, 0;...
    0, 0, 0;...
    1, 0, 1;...
    0, 1, 1;...
    1, 1, 0];
  handles.sim = [];
  guidata(fig, handles);
  
  if nargout > 0
    varargout{1} = fig;
  end
  
elseif ischar(varargin{1}) % INVOKE NAMED SUBFUNCTION OR CALLBACK
  
  try
    if (nargout)
      [varargout{1:nargout}] = feval(varargin{:}); % FEVAL switchyard
    else
      feval(varargin{:}); % FEVAL switchyard
    end
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
%| tags as fieldnames, e.g. handles.figure1, handles.slPhase. This
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

function varargout = figure1_DeleteFcn(h, ~, handles, varargin)
[~,name] = fileparts(get(handles.hh, 'FileName'));
eval([name '(''DeletePlugins'', handles.figure1, handles.hh)']);

% --------------------------------------------------------------------
function FileUpdate(hObject, handles)
FitData(handles.FitMethod, handles);

% --------------------------------------------------------------------
function varargout = pushbutton1_Callback(h, eventdata, handles, varargin)

% --------------------------------------------------------------------
function varargout = lbPars_Callback(h, eventdata, handles, varargin)
Res = get(handles.lbPars, 'String');
val = get(handles.lbPars, 'Value');

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

Res = get(handles.lbPars, 'String');
val = get(handles.lbPars, 'Value');
newval = strtrim(get(handles.ePars, 'String'));

[name, ~] = strtok(Res{val}, '=');
name = strtrim(name);

switch get(handles.ePars, 'UserData')
  case 'function'
    set(handles.ePars, 'String', newval);
    
    tokens = kv_autofit([name ' = ' newval]);
    
    %%%%%%%%%%%%% load script %%%%%%%%%%%%%%%%%%%%%%
    for k=1:size(Res,1)
      if ~strcmp(strtrim(strtok(Res{k},'=')),'f')
        try eval([Res{k} ';']);catch;end
      end
    end
    tempcell = Res;
    Res = {[name ' = ' newval]};
    for k = 1: size(tokens, 2)
      try var = eval(tokens{k}); catch err, var = 0; end
      Res{end+1} = [tokens{k},' = ', num2str(var)];
    end
    %%%% find strings with additional parameters %%%
    for count = 1:length(tempcell)
      if length(tempcell{count})>6
        if strcmp(tempcell{count}(1:6), 'FitRng')
          Res{end+1} = tempcell{count};
        end
      end
    end
    clear tempcell;
  case 'string'
    set(handles.ePars, 'string', newval);
    Res{val} = [name ' = ''', newval,''''];
  otherwise
    nums = str2num(newval);
    set(handles.ePars, 'String', num2str(nums));
    Res{val} = [name ' = ' num2str(nums)];
end
set(handles.lbPars, 'String', Res);
lbPars_Callback(h, eventdata, handles);
handles.Script = Res;
guidata(handles.figure1, handles);

if strcmp(get(handles.ePars, 'UserData'), 'value'),
  DrawAll(handles);
end

% --------------------------------------------------------------------
function varargout = slPars_Callback(h, eventdata, handles, varargin)
parname = get(handles.lbPars, 'String');
parval = get(handles.lbPars, 'Value');
if ~strcmp(get(handles.ePars, 'UserData'),'value'), return; end;

str = get(handles.pmPars, 'String');

shift = get(handles.slPars, 'Value');
set(handles.slPars, 'Value', 0);
num = get(handles.pmPars, 'Value');

dim = str2num([str{num}]);

CurVal = str2double(get(handles.ePars, 'String'));
Val = CurVal + dim*shift;

set(handles.ePars, 'String', num2str(Val));
ePars_Callback(h, eventdata, handles, varargin);

% --------------------------------------------------------------------
function varargout = pmPars_Callback(h, ~, handles, varargin)

% --------------------------------------------------------------------
function varargout = pbIntegral_Callback(~, ~, handles, varargin)
[ax,y,area] = GetData(handles);
dx = mean(diff(ax.x));

res = sum(y(area))*dx;
set(handles.eResInt, 'String', sprintf('INT r=%5.3f , i=%5.3f', real(res), imag(res)));

% --------------------------------------------------------------------
function varargout = edit2_Callback(~, ~, handles, varargin)


% --------------------------------------------------------------------
function varargout = btnFit_Callback(h, ~, handles, varargin)

[ax,y,areafit] = GetRawData(handles);
x = ax.x;

[strings,handles.sim.y]= kv_autofit(get(handles.lbPars, 'String'), x, real(y), areafit);
set(handles.lbPars, 'String', strings);
handles.Script = strings;
guidata(handles.figure1, handles);

try
  set(handles.lbPars, 'String', strings);
catch
  set(handles.lbPars, 'String', strings, 'Value', 1);
end
lbPars_Callback(h, [], handles);

num = get(handles.pmPars, 'Value');
handles.Script{num} = strings;
guidata(handles.figure1, handles);

FitData(handles.FitMethod, handles);

%--------------------------------------------------------------------
function evallist(handles)

[viewer_projection, viewer_range] = getslider_kazan(handles);
str = get(handles.lbPars, 'String');
val = 0;
hand = guidata(handles.hh);
FitRng = [];
eval(['x = hand.src.ax.', viewer_projection, ';']);
eval(['y = hand.src.y', viewer_range, ';']);
[~, ind] = max(size(y));
if ind == 2, x = x.'; end
for count = 1:size(str, 1)
  if ~strcmp(str{count}(1:3), 'f =')
    eval([str{count}, ';']);
  else
    val = count;
  end
end

[handles.range(handles.ActivePage).pnts, handles.range(handles.ActivePage).idx] = getrange(FitRng, x);
handles.range(handles.ActivePage).bracket = FitRng;

if ~val, disp('EVALLIST: Can not find function'); return; end
if findstr(str{val}, 'polyfit')
  btnFit_Callback(handles.btnFit, [], handles)
else
  eval([str{val}, ';']);
  if strcmp(viewer_range, '(:, :)')
    switch viewer_projection
      case 'x'
        eval('handles.sim.y = f(:, ones(size(hand.src.y, 2), 1));');
      case 'y'
        eval('handles.sim.y = f(ones(size(hand.src.y, 1), 1), :);');
    end
  else
    eval(['handles.sim.y', viewer_range, ' = f;']);
  end
  guidata(handles.figure1, handles);
end

% --------------------------------------------------------------------
function [points,FitRng] = getrange(FitRngX, x)
numpoints=length(x);
pnts = zeros(1,numpoints);
sz = size(FitRngX);
allranges = reshape(FitRngX, 1, prod(sz));
[~,idx] = min(abs(allranges(ones(numpoints,1),:)-x(:,ones(1,prod(sz)))));
FitRng = reshape(idx, sz);
if ~isempty(FitRng)
  for i = 1:size(FitRng, 1)
    pnts([min(FitRng(i, 1), end):min(FitRng(i, 2), end)]) = 1;
  end
  points = find(pnts);
else
  points = [];
end

% --------------------------------------------------------------------
function varargout = cbFullIntRange_Callback(h, eventdata, handles, varargin)
eFirstPos_Callback(h, eventdata, handles, varargin)
% --------------------------------------------------------------------
function varargout = eLastPos_Callback(h, eventdata, handles, varargin)
eFirstPos_Callback(h, eventdata, handles, varargin);

% --------------------------------------------------------------------
function varargout = eFirstPos_Callback(h, ~, handles, varargin)
hand = guidata(handles.hh);
y = hand.src.y;

if get(handles.cbFullIntRange,'Value')
  handles.area = 1:size(y,1);
else
  str1 = get(handles.eFirstPos, 'String');
  str2 = get(handles.eLastPos, 'String');
  p1=str2double(str1);
  p2=str2double(str2);
  if isempty(p1), p1=hand.src.ax.x(1); end
  if isempty(p2), p2=hand.src.ax.x(size(handles.sim.y,1)); end
  [~,idx1]=min(abs(hand.src.ax.x-p1));
  [~,idx2]=min(abs(hand.src.ax.x-p2));
%   set(handles.eFirstPos, 'String', num2str(hand.src.ax.x(idx1)));
%   set(handles.eLastPos, 'String', num2str(hand.src.ax.x(idx2)));
  handles.area = idx1:idx2;
end

guidata(handles.figure1, handles);
DrawAll(handles);

% --------------------------------------------------------------------
function varargout = mContextShowRange_Callback(h, ~, handles, varargin)
DrawAll(handles);
% --------------------------------------------------------------------
function varargout = cbShowIntRange_Callback(h, ~, handles, varargin)
DrawAll(handles)
% --------------------------------------------------------------------
function varargout = cbShift_Callback(h, ~, handles, varargin)
DrawAll(handles);
% --------------------------------------------------------------------
function varargout = cbSquare_Callback(h, ~, handles, varargin)
DrawAll(handles);

% --------------------------------------------------------------------
function varargout = pbPages_Callback(h, eventdata, handles, varargin)
handles.Pages{handles.ActivePage} = get(handles.lbPars,'String');
hh = [handles.pb0,handles.pb1,handles.pb2,handles.pb3,handles.pb4,handles.pb5];
set(hh, 'Value',0)
set(hh(eventdata+1), 'Value',1)
handles.ActivePage = eventdata+1;
the_script = handles.Pages{eventdata+1};

if length(the_script) == 1
  tokens = kv_autofit(the_script{1});
  for k = 1: size(tokens, 2)
    try var = eval(tokens{k}); catch, var = 0; end
    the_script{end+1} = [tokens{k},' = ', num2str(var)];
  end
  handles.Pages{eventdata+1} = the_script;
end
set(handles.lbPars,'String', the_script,'Value',1);
guidata(handles.figure1, handles);
FitData(handles.FitMethod, handles);

lbPars_Callback(h, eventdata, handles);

% --------------------------------------------------------------------
function [prj, sel] = getslider_kazan(handles)
hand = guidata(handles.hh);
str = {'x', 'y'};
prj = str{hand.Projection};
if hand.PlotType ==1
  switch hand.Projection
    case 1
      sel = ['(:, ','min(',  num2str(hand.Selection),', end))'];
    case 2
      sel = ['(min(', num2str(hand.Selection),', end), :)'];
  end
else
  sel = '(:, :)';
end

%--------------------------------------------------------------------
function pbFitLW_Callback(hObject, ~, handles)

switch hObject
  case handles.pbFitLW_L, handles.FitMethod = 'FitLW_L';
  case handles.pbFitLW_V, handles.FitMethod = 'FitLW_V';
  case handles.pbFitLW_G, handles.FitMethod = 'FitLW_G';
  case handles.pbFitLW_Mod, handles.FitMethod = 'FitLW_Mod';
  case handles.pbFitLWHH, handles.FitMethod = 'FitLWHH';
  case handles.pbFitLWpp, handles.FitMethod = 'FitLWpp';
  otherwise, handles.FitMethod = '';
end

guidata(handles.figure1, handles);
FitData(handles.FitMethod, handles);

% --------------------------------------------------------------------
function pbPhase_Callback(hObject, ~, handles)
FitData(handles.FitMethod, handles)

% --------------------------------------------------------------------
function ePhase_Callback(hObject, ~, handles)
CurVal=eval(get(handles.ePhase, 'String'));
set(handles.ePhase, 'String', num2str(CurVal));
FitData(handles.FitMethod, handles)

% --------------------------------------------------------------------
function slPhase_Callback(hObject, ~, handles)
shift =get(handles.slPhase, 'Value');
set(handles.slPhase, 'Value',0);
CurVal=str2double(get(handles.ePhase, 'String'));
cc = safeget(handles, 'PhaseFactor',1.);
Val = max(min(CurVal + shift*cc, 180), -180);
set(handles.ePhase, 'String', num2str(Val))
FitData(handles.FitMethod, handles)

% --------------------------------------------------------------------
function pb01_Callback(hObject, ~, handles)
if get(handles.pb01, 'Value'), handles.PhaseFactor = 0.1; set(handles.pb10, 'Value',0); 
else handles.PhaseFactor = 1; end
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function pb10_Callback(hObject, ~, handles)
if get(handles.pb10, 'Value'), handles.PhaseFactor = 10; set(handles.pb01, 'Value',0); 
else handles.PhaseFactor = 1; end
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function pbBaseline_Callback(hObject, ~, handles)
FitData(handles.FitMethod, handles)

% --------------------------------------------------------------------
function y = simple_fit(a, xdata, xpatt, pattern, derivative)

lw = lshape(xpatt, 0, abs(a(3)), derivative, 0);
y = real(conv2(lw(:,1),pattern,'same'));
y = a(1) * interp1(xpatt + a(2), y, xdata);

%--------------------------------------------------------------------
%--------------------------------------------------------------------
%-------------- D R A W    A L L ------------------------------------
%--------------------------------------------------------------------
%--------------------------------------------------------------------

function DrawAll(handles)
[ax,~,area_fit] = GetRawData(handles);

xx = handles.dest.x;
yy = handles.dest.y;

if get(handles.cbFullIntRange,'Value')
  area_integral = 1:size(xx,1);
else
  str1 = get(handles.eFirstPos, 'String');
  str2 = get(handles.eLastPos, 'String');
  p1=str2double(str1);
  p2=str2double(str2);
  if isempty(p1), p1=xx(1); end
  if isempty(p2), p2=xx(size(yy,1)); end
  [~,idx1]=min(abs(xx-p1));
  [~,idx2]=min(abs(xx-p2));
  area_integral = idx1:idx2;
end

% Plot data
hand = GetHandle(handles);
% hand.out{1}.ax = ax;
% hand.out{1}.ax.Color = [0,0,1];
% hand.out{1}.ax.s = 0;
% hand.out{1}.y = y;
% hand.out{1}.title = 'Src';

isSquare = get(handles.cbSquare, 'Value');
isShift = get(handles.cbShift,   'Value');


if isSquare,  yy=abs(yy); end
if isShift
  shift = min(real(yy(area_integral))) + 1i*min(imag(yy(area_integral)));
  yy = yy - shift;
end

% Processed dataset
hand.out = {};
hand.out{end+1}.ax = ax;
hand.out{end}.y = yy;
hand.out{end}.ax.Color = [0, 0, 1]; % always blue
hand.out{end}.ax.s = 0;
hand.out{end}.title = 'Processed';

% fitting range
if get(handles.pbBaseline, 'value') && any(area_fit == false)
  hand.out{end+1}.ax = ax;
  hand.out{end}.ax.x = ax.x(area_fit);
  hand.out{end}.y = yy(area_fit);
  hand.out{end}.ax.Color = [0, 0, 0]; % always green
  hand.out{end}.ax.s = 0;
  hand.out{end}.ax.Marker = 'o';
  hand.out{end}.ax.LineStyle = 'none';
  hand.out{end}.title = 'fit area';
end

% Draw area
if get(handles.cbShowIntRange, 'Value')
  showarea = fix(linspace(min(area_integral), max(area_integral), 50*length(area_integral)/length(yy)));
  hand.out{end+1}.ax = ax;
  hand.out{end}.ax.x=[];
  hand.out{end}.ax.x(1,:) = ax.x(showarea);
  my1 = min(real(yy(:))); my2 = max(real(yy(:)));
  k = ones(size(yy))*(my1 + 0.1*(my2-my1));
  hand.out{end}.y = k(showarea);
  hand.out{end}.ax.Color = [0.7, 0, 0]; % always gray
  hand.out{end}.ax.diff = '';
  hand.out{end}.ax.filt = '';
  hand.out{end}.ax.s = 0;
  hand.out{end}.title = 'Area';
end

if ~isempty(handles.fit.y)
  hand.out{end+1}.ax = handles.fit.ax;
  hand.out{end}.ax.x = handles.fit.x;
  hand.out{end}.ax.y = [1];
  hand.out{end}.y = handles.fit.y;
  hand.out{end}.ax.Color = [.6, .2, .0]; % always gray
  hand.out{end}.ax.diff = '';
  hand.out{end}.ax.filt = '';
  hand.out{end}.ax.s = 0;
  hand.out{end}.title = 'Area';
end

guidata(handles.hh, hand);
[~,name] = fileparts(get(handles.hh, 'FileName'));
eval([name '(''SetDataSource'', 2, hand)']);

%--------------------------------------------------------------------
function [ax,y,areafit] = GetRawData(handles)
hand = guidata(handles.hh);

if strcmp(handles.src.source, 'output')
  ax = handles.src.ax;
  y = handles.src.y;
else
  ax = hand.src.ax;
  y = hand.src.y;
end

if length(handles.AreaFit) ~= length(y)
  areafit = true(length(y), 1);
else
  areafit = handles.AreaFit;
end

%--------------------------------------------------------------------
function [ax,y,area,areafit] = GetData(handles)
[ax,y,areafit] = GetRawData(handles);
x = ax.x;

if get(handles.tbFilterOn, 'value')
  switch handles.src.filter
    case 'savgol1', y = sgolayfilt(y, 1, 7);
    case 'savgol2', y = sgolayfilt(y, 1, 13);
    case 'savgol3', y = sgolayfilt(y, 1, 35);
    case 'savgol4', y = sgolayfilt(y, 1, 53);
  end
end

% subtract baseline
if get(handles.pbBaseline, 'value')
  the_script = get(handles.lbPars, 'string');

  for ii=length(the_script):-1:1
    eval([the_script{ii},';']);
  end
  y = y - f;  
end

if strcmp(handles.src.type, 'absorption') && strcmp(handles.dest.type, 'integral')
  y = cumsum(y);
elseif strcmp(handles.src.type, 'derivative') && strcmp(handles.dest.type, 'absorption')
  y = cumsum(y);
elseif strcmp(handles.src.type, 'absorption') && strcmp(handles.dest.type, 'derivative')
  y = gradient(y, mean(diff(x)));
elseif strcmp(handles.src.type, 'derivative') && strcmp(handles.dest.type, 'integral')
  y = cumsum(cumsum(y));
end

if get(handles.cbFullIntRange,'Value')
  area = 1:size(x,1);
else
  str1 = get(handles.eFirstPos, 'String');
  str2 = get(handles.eLastPos, 'String');
  p1=str2double(str1);
  p2=str2double(str2);
  if isempty(p1), p1=x(1); end
  if isempty(p2), p2=x(size(y,1)); end
  [~,idx1]=min(abs(x-p1));
  [~,idx2]=min(abs(x-p2));
  area = idx1:idx2;
end

%--------------------------------------------------------------------
function hand = GetHandle(handles)
hand = guidata(handles.hh);

%--------------------------------------------------------------------
function tbDest_Callback(hObject, eventdata, handles)
btns = [handles.tbDestAb, handles.tbDestDer, handles.tbDestInt];
val = {'absorption', 'derivative', 'integral'};
if any(btns == hObject)
  set(btns(btns ~= hObject), 'value', 0);
  set(hObject, 'value', 1);
  handles.dest.type = val(btns == hObject);
  guidata(hObject, handles);
  FitData(handles.FitMethod, handles)
end

%--------------------------------------------------------------------
function tbSrc_Callback(hObject, ~, handles)
btns = [handles.tbSrcAb, handles.tbSrcDer];
val = {'absorption', 'derivative'};
if any(btns == hObject)
  set(btns(btns ~= hObject), 'value', 0);
  set(hObject, 'value', 1);
  handles.src.type = val(btns == hObject);
  guidata(hObject, handles);
  FitData(handles.FitMethod, handles)
end

%--------------------------------------------------------------------
function tbSrcOutput_Callback(hObject, ~, handles)
btns = [handles.tbSrcSource, handles.tbSrcOutput];
val = {'source', 'output'};
if any(btns == hObject)
  set(btns(btns ~= hObject), 'value', 0);
  set(hObject, 'value', 1);
  handles.src.source = val(btns == hObject);
  
  if strcmp(handles.src.source, 'output')
      hand = guidata(handles.hh);
      handles.src.ax = hand.out{1}.ax;
      handles.src.y = hand.out{1}.y;
  end
  guidata(hObject, handles);
  FitData(handles.FitMethod, handles)
end

%--------------------------------------------------------------------
function tbFilterOn_Callback(hObject, ~, handles)
btns = [handles.tbFilter1, handles.tbFilter2, handles.tbFilter3, handles.tbFilter4];
val = {'savgol1', 'savgol2', 'savgol3','savgol4'};
if any(btns == hObject)
  set(btns(btns ~= hObject), 'value', 0);
  set(hObject, 'value', 1);
  handles.src.filter = val{btns == hObject};
  
  guidata(hObject, handles);
end
FitData(handles.FitMethod, handles)

%--------------------------------------------------------------------
function btnGetMinRange_Callback(hObject, eventdata, handles)
hand = guidata(handles.hh);
set(0, 'currentfigure', hand.MainFigure);
pos = ginput(1);
set(handles.eFirstPos, 'string', num2str(pos(1)));
eFirstPos_Callback(hObject, eventdata, handles);


%--------------------------------------------------------------------
function btnGetMaxRange_Callback(hObject, eventdata, handles)
hand = guidata(handles.hh);
set(0, 'currentfigure', hand.MainFigure);
pos = ginput(1);
set(handles.eLastPos, 'string', num2str(pos(1)));
eFirstPos_Callback(hObject, eventdata, handles);

%--------------------------------------------------------------------
function btnAddFitRange_Callback(hObject, eventdata, handles)
hand = guidata(handles.hh);
set(0, 'currentfigure', hand.MainFigure);
disp('Select the beginning and the end of the range');
pos = ginput(2); pos1 = sort(pos(:,1));

[ax,~,~,areafit] = GetData(handles);
areafit(ax.x > pos1(1) & ax.x < pos1(2)) = true;
handles.AreaFit = areafit;
guidata(hObject, handles);
FitData(handles.FitMethod, handles)

%--------------------------------------------------------------------
function btnRemoveFitRange_Callback(hObject, ~, handles)
hand = guidata(handles.hh);
set(0, 'currentfigure', hand.MainFigure);
disp('Select the beginning and the end of the range');
pos = ginput(2); pos1 = sort(pos(:,1));

[ax,~,~,areafit] = GetData(handles);
areafit(ax.x > pos1(1) & ax.x < pos1(2)) = false;
handles.AreaFit = areafit;
guidata(hObject, handles);
FitData(handles.FitMethod, handles)

%--------------------------------------------------------------------
function eSpinProbe_Callback(hObject, ~, handles)

%--------------------------------------------------------------------
function eSpinProbe_CreateFcn(hObject, ~, handles)

%--------------------------------------------------------------------
function figure1_SizeChangedFcn(~, ~, handles)
standard_size = [0.144, 0.528, 0.238, 0.321];
fig_size = get(handles.figure1, 'Position');
scaling_factor = fig_size(4) / standard_size(4);
DefaultFont = handles.DefaultFont;
if isfield(handles, 'hh')
   hand = guidata(handles.hh);
   DefaultFont = safeget(hand, 'DefaultFont', 8);
end
font_size = DefaultFont + (scaling_factor-1) * (DefaultFont-2);
kazan_size_change(handles.figure1, font_size)


%--------------------------------------------------------------------
function FitData(processing, handles)
[ax,yy,areafit] = GetRawData(handles);
xx = ax.x;

% source data verification
is_src_derivative = false;
x_estimate  = 0;
lw_estimate = (max(xx)-min(xx)) / 5.0;
if strcmp(handles.src.type, 'derivative')
  is_src_derivative = true;
  [~,idxmin] = min(yy);
  [~,idxmax] = max(yy);
  x_estimate   = (xx(idxmin)+xx(idxmax)) / 2;
  lw_estimate  = abs(xx(idxmax)-xx(idxmin)); % LWpp
elseif strcmp(handles.src.type, 'absorption')
  [~,idxmax] = max(yy);
  x_estimate = xx(idxmax);
end

% base line correction
if handles.pbBaseline.Value
  [strings,handles.sim.y]= kv_autofit(get(handles.lbPars, 'String'), ax.x,  real(yy), areafit);
  set(handles.lbPars, 'String', strings);
  handles.Script = strings;
  guidata(handles.figure1, handles);
  
  try
    set(handles.lbPars, 'String', strings);
  catch
    set(handles.lbPars, 'String', strings, 'Value', 1);
  end
  lbPars_Callback([], [], handles);
  num = get(handles.pmPars, 'Value');
  handles.Script{num} = strings;
  guidata(handles.figure1, handles)

  yy = yy - handles.sim.y;
end

% phase correction
if get(handles.pbPhase, 'value')
  phase = str2double(get(handles.ePhase, 'String'));
  yy = real(yy.*exp(-1i*phase/180*pi));
end

% convert to output representation
if strcmp(handles.src.type, 'absorption') && strcmp(handles.dest.type, 'integral')
  yy = cumsum(yy);
elseif strcmp(handles.src.type, 'derivative') && strcmp(handles.dest.type, 'absorption')
  yy = cumsum(yy); is_src_derivative = false;
elseif strcmp(handles.src.type, 'absorption') && strcmp(handles.dest.type, 'derivative')
  yy = gradient(yy, mean(diff(x))); is_src_derivative = true;
elseif strcmp(handles.src.type, 'derivative') && strcmp(handles.dest.type, 'integral')
  yy = cumsum(cumsum(yy));
end

% filter data
if get(handles.tbFilterOn, 'value')
  switch handles.src.filter
    case 'savgol1', yy = sgolayfilt(yy, 1, 5);
    case 'savgol2', yy = sgolayfilt(yy, 1, 11);
    case 'savgol3', yy = sgolayfilt(yy, 1, 33);
    case 'savgol4', yy = sgolayfilt(yy, 1, 53);
  end
end

handles.dest.x = xx;
handles.dest.y = yy;

switch processing
  case 'FitLW_L'
    predicted = @(a, xdata) abs(a(1))*lshape(xdata, a(2), abs(a(3)), is_src_derivative, 0); % lorentian
    
    opts = optimset('Display','off');
    [ahat] = lsqcurvefit(predicted,[max(yy), x_estimate, 0.3],xx,yy,[],[],opts);
    [ahat] = lsqcurvefit(predicted,ahat,xx,yy,[],[],opts);
    
    handles.fit.ax = ax;
    handles.fit.x = ax.x;
    handles.fit.y = predicted(ahat, ax.x);
    the_error = std(yy-handles.fit.y);
    report_message = sprintf('LW-L = %5.4f\nSD=%5.4f', abs(ahat(3)), the_error);
  case 'FitLW_V'
    sigma = str2double(handles.eFitLW_V.String); % [G]
    if sigma <= 0
      predicted = @(a, xdata) abs(a(1))*voigtian(xdata,a(2),abs([a(3),a(4)]),is_src_derivative);
      opts = optimset('Display','off');
      [ahat] = lsqcurvefit(predicted,[max(yy), x_estimate, [lw_estimate/2, lw_estimate/2]],xx,yy,[],[],opts);
      [ahat] = lsqcurvefit(predicted,ahat,xx,yy,[],[],opts);

      handles.fit.ax = ax;
      handles.fit.x = ax.x;
      handles.fit.y = predicted(ahat, ax.x);
      the_error = std(yy-handles.fit.y);
      report_message = sprintf('LW-GL = %5.4f / %5.4f\nSD=%5.4f', abs(ahat(3)), abs(ahat(4)), the_error);
    else
      predicted = @(a, xdata) abs(a(1))*voigtian(xdata,a(2),abs([sigma,a(3)]),is_src_derivative);
      opts = optimset('Display','off');
      [ahat] = lsqcurvefit(predicted,[max(yy), x_estimate, max(abs(sigma-lw_estimate), lw_estimate/4)],xx,yy,[],[],opts);
      [ahat] = lsqcurvefit(predicted,ahat,xx,yy,[],[],opts);

      handles.fit.ax = ax;
      handles.fit.x = ax.x;
      handles.fit.y = predicted(ahat, ax.x);
      the_error = std(yy-handles.fit.y);
      report_message = sprintf('LW-GL = %5.4f / %5.4f\nSD=%5.4f', sigma, abs(ahat(3)), the_error);
    end
  case 'FitLW_G'
    predicted = @(a, xdata) a(1)*lshape(xdata, a(2), abs(a(3)), is_src_derivative, 1); % gaussian
    
    opts = optimset('Display','off');
    [ahat] = lsqcurvefit(predicted,[max(yy), x_estimate, lw_estimate],xx,yy,[],[],opts);
    [ahat] = lsqcurvefit(predicted,ahat,xx,yy,[],[],opts);
    
    handles.fit.ax = ax;
    handles.fit.x = ax.x;
    handles.fit.y = predicted(ahat, ax.x);
    the_error = std(yy-handles.fit.y);
    report_message = sprintf('LW-G = %5.4f\nSD=%5.4f', abs(ahat(3)), the_error);
  case 'FitLW_Mod'
    spin_probe = handles.eSpinProbe.String{handles.eSpinProbe.Value};
    [~, shf_pars] = cw_shf_model(spin_probe);
    [pattern, resolution] = cw_shf(shf_pars);
    Bshf = (1:length(pattern))' * resolution; Bshf = Bshf - mean(Bshf);
    predicted = @(a, xdata) simple_fit(a, xdata, Bshf, pattern, is_src_derivative);

    opts = optimset('Display','off');
    [ahat] = lsqcurvefit(predicted,[max(yy), x_estimate, lw_estimate/2],xx,yy,[],[],opts);
    [ahat] = lsqcurvefit(predicted,ahat,xx,yy,[],[],opts);

    handles.fit.ax = ax;
    handles.fit.x = ax.x;
    handles.fit.y = predicted(ahat, ax.x);
    the_error = std(yy-handles.fit.y);
    report_message = sprintf('LW-L = %5.4f\nSD=%5.4f', abs(ahat(3)), the_error);
  case 'FitLWHH'
    if is_src_derivative
      report_message = 'LWhh = ?';
    else
      % locate maximum
      [mm,pidx] = max(yy);
      
      for ii=pidx:-1:1
        if yy(ii) < mm/2
          lpidx=ii; break;
        end
      end
      for ii=pidx:length(ax.x)
        if yy(ii) < mm/2
          rpidx=ii; break;
        end
      end
      
      handles.fit.ax = ax;
      handles.fit.x = [xx(lpidx),xx(lpidx),xx(rpidx),xx(rpidx)];
      handles.fit.y = mm*[0.1,0,0,0.1];
      report_message = sprintf('LWhh = %5.4f', xx(rpidx)-xx(lpidx));
    end
  case 'FitLWpp'
    if is_src_derivative
      % locate maximum
      [~,lpidx] = min(yy);
      [~,rpidx] = max(yy);
      
      handles.fit.ax = ax;
      mp = max(yy);
      handles.fit.x = [xx(lpidx),xx(lpidx),xx(rpidx),xx(rpidx)];
      handles.fit.y = mp*[0.1,0,0,0.1];
      report_message = sprintf('LWpp = %5.4f', abs(xx(rpidx)-xx(lpidx)));
    else
      report_message = 'LWhh = ?';
    end
  otherwise
    handles.fit.x = [];
    handles.fit.y = [];
    report_message = '?';
end

guidata(handles.figure1, handles)
DrawAll(handles);
set(handles.eResInt, 'String', report_message);


