function varargout = transient(varargin)
% TRANSIENT Application M-file for transient.fig
%    FIG = TRANSIENT launch transient GUI.
%    TRANSIENT('callback_name', ...) invoke the named callback.
% Contact: bepel@uchicago.edu, aus40@psu.edu

% Last Modified by GUIDE v2.5 28-Aug-2023 23:04:41

if nargin == 0  % LAUNCH GUI

	fig = openfig(mfilename,'reuse');

	% Use system color scheme for figure:
	set(fig,'Color',get(0,'defaultUicontrolBackgroundColor'));

	% Generate a structure of handles to pass to callbacks, and store it. 
	handles = guihandles(fig);
  handles.AreaFit = [];
  handles.TimeField = 1;
  handles.DefaultFont = 8;
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

% --------------------------------------------------------------------
function varargout = Transient_DeleteFcn(~, ~, handles, varargin)
[~,name] = fileparts(get(handles.hh, 'FileName'));
eval([name '(''DeletePlugins'', handles.Transient, handles.hh)']);

% --------------------------------------------------------------------
function FileUpdate(~, handles)
[~,y] = GetRawData(handles);
handles.eInfo.String = sprintf("field %i pts\n time %i pts", size(y,1), size(y,2));

% --------------------------------------------------------------------
function pushbutton1_Callback(~, ~, handles, varargin)
FileUpdate([], handles);
DrawAll(handles);

% --------------------------------------------------------------------
function DrawAll(handles)
hand = GetHandle(handles);
[ax,y,areafit] = GetRawData(handles);

DARKstart = get(handles.eBLstart, 'String');
DARKend = get(handles.eBLend, 'String');
Sstart = get(handles.eSstart, 'String');
Send = get(handles.eSend, 'String');
% Lstart = get(handles.eLstart, 'String');
% Lend = get(handles.eLend, 'String');

SIG = str2double(Sstart):str2double(Send);
DARK = str2double(DARKstart):str2double(DARKend);
SIG = SIG(SIG > 0 & SIG <= size(y,2));
DARK = DARK(DARK > 0 & DARK <= size(y,2));

if ~isempty(DARK)
  MEANDARK  = mean(y(:, DARK), 2);
  y = y - repmat(MEANDARK, 1, size(y,2));
end
ysm = y;

if handles.tbS1.Value
  for jj=1:size(y,1)
    ysm(jj,:) = datasmooth(y(jj,:),5,'savgol',2); % smooth the data in the time dimension 
  end
elseif handles.tbS2.Value
  for jj=1:size(y,1)
    ysm(jj,:) = datasmooth(y(jj,:),10,'savgol',2); % smooth the data in the time dimension 
  end
elseif handles.tbS3.Value
  for jj=1:size(y,1)
    ysm(jj,:) = datasmooth(y(jj,:),20,'savgol',2); % smooth the data in the time dimension 
  end
end

if get(handles.cbInvert,'Value')==1
  y = -y;
  ysm = -ysm;
end

hand.out = {};
if handles.TimeField == 1

  if isempty(SIG) || handles.tbShowAllTraces.Value > 0.5
    
    yout = y;
    
    BL = zeros(size(y));
    for ii=1:size(y,2)
      if numel(areafit) > 3
        if handles.tbF0.Value
          BL(:,ii) = mean(yout(areafit,ii), 1);
        elseif handles.tbF1.Value
          [P] = polyfit(ax.x(areafit), yout(areafit,ii), 1);
          BL(:,ii) = polyval(P, ax.x);
        elseif handles.tbF2.Value
          [P] = polyfit(ax.x(areafit), yout(areafit,ii), 2);
          BL(:,ii) = polyval(P, ax.x);
        elseif handles.tbF3.Value
          [P] = polyfit(ax.x(areafit), yout(areafit,ii), 3);
          BL(:,ii) = polyval(P, ax.x);
        end
      end
    end
  
    hand.out{1}.ax = ax;
    hand.out{1}.y = yout - BL;
    hand.out{1}.title = 'original';

    % fitting area
    hand.out{end+1}.ax = ax;
    hand.out{end}.ax.x = ax.x(areafit);
    hand.out{end}.y = yout(areafit,:) - BL(areafit,:);
    hand.out{end}.ax.Color = [0, 0, 0]; % always green
    hand.out{end}.ax.s = 0;
    hand.out{end}.ax.Marker = 'o';
    hand.out{end}.ax.LineStyle = 'none';
    hand.out{end}.title = 'fit area';
    
    guidata(handles.hh, hand);
    [~,name] = fileparts(get(handles.hh, 'FileName'));
    eval([name '(''SetDataSource'', 2, hand)']);
    return;
  end
  
  ysmout = sum(ysm(:, SIG), 2)/length(SIG);
  yout = sum(y(:, SIG), 2)/length(SIG);
  
  if handles.tbS1.Value
      ysmout = datasmooth(ysmout,1,'savgol',1);
  elseif handles.tbS2.Value
      ysmout = datasmooth(ysmout,3,'savgol',1);
  elseif handles.tbS3.Value
      ysmout = datasmooth(ysmout,5,'savgol',1);
  end

  BL = zeros(size(ax.x));
  if numel(areafit) > 3
    if handles.tbF0.Value
      BL = mean(yout(areafit))*ones(size(ax.x));
    elseif handles.tbF1.Value
      [P] = polyfit(ax.x(areafit), ysmout(areafit), 1);
      BL = polyval(P, ax.x);
    elseif handles.tbF2.Value
      [P] = polyfit(ax.x(areafit), ysmout(areafit), 2);
      BL = polyval(P, ax.x);
    elseif handles.tbF3.Value
      [P] = polyfit(ax.x(areafit), ysmout(areafit), 3);
      BL = polyval(P, ax.x);
    end
  end

  if handles.cbShowFit.Value
    hand.out = {};
    hand.out{1}.ax = ax;
    hand.out{1}.ax.y = 0;
    hand.out{1}.ax.ylabel = '';
    hand.out{1}.y = ysmout;
    hand.out{1}.title = 'smoothed';
    
    if handles.cbShowSrc.Value
      hand.out{end+1}.ax = ax;
      hand.out{end}.y = yout;
      hand.out{end}.ax.Color = [1, 0, 0]; % always red
      hand.out{end}.title = 'original';
    end
    
    % fitting area
    hand.out{end+1}.ax = ax;
    hand.out{end}.ax.x = ax.x(areafit);
    hand.out{end}.y = ysmout(areafit);
    hand.out{end}.ax.Color = [0, 0, 0]; % always green
    hand.out{end}.ax.s = 0;
    hand.out{end}.ax.Marker = 'o';
    hand.out{end}.ax.LineStyle = 'none';
    hand.out{end}.title = 'fit area';
    
    % baseline
    hand.out{end+1}.ax = ax;
    hand.out{end}.ax.x = ax.x;
    hand.out{end}.y = BL;
    hand.out{end}.ax.Color = [0, 1, 0]; % always green
    hand.out{end}.ax.s = 0;
    hand.out{end}.ax.LineStyle = '-';
    hand.out{end}.title = 'baseline';
    
  else
    hand.out = {};
    hand.out{1}.ax = ax;
    hand.out{1}.y = ysmout - BL;
    hand.out{1}.title = 'baseline corrected';

    if handles.cbShowSrc.Value
      hand.out{end+1}.ax = ax;
      hand.out{end}.y = yout - BL;
      hand.out{end}.ax.Color = [1, 0, 0]; % always red
      hand.out{end}.title = 'original';
    end
    
    % fitting area
    hand.out{end+1}.ax = ax;
    hand.out{end}.ax.x = ax.x(areafit);
    hand.out{end}.y = ysmout(areafit) - BL(areafit);
    hand.out{end}.ax.Color = [0, 0, 0]; % always green
    hand.out{end}.ax.s = 0;
    hand.out{end}.ax.Marker = 'o';
    hand.out{end}.ax.LineStyle = 'none';
    hand.out{end}.title = 'fit area';
  end
else   % Time domain
  if isempty(SIG)
    
    yout = y;
    hand.out{end+1}.ax = ax;
    hand.out{end}.ax.x = ax.y;
    hand.out{end}.ax.xlabel = ax.ylabel;
    hand.out{end}.ax.y = ax.x;
    hand.out{end}.ax.ylabel = ax.ylabel;
    hand.out{end}.y = yout';
    hand.out{end}.title = 'time trace';
    
    if numel(DARK) > 0
      hand.out{end+1}.ax = ax;
      hand.out{end}.ax.x = ax.y(DARK);
      hand.out{end}.ax.y = ax.x;
      hand.out{end}.ax.xlabel = ax.ylabel;
      hand.out{end}.y = yout(:,DARK)';
      hand.out{end}.ax.Color = [0, 0, 0];
      hand.out{end}.ax.Marker = 'o';
      hand.out{end}.ax.LineStyle = 'none';
      hand.out{end}.title = 'dark';
    end
  else
    ShowFieldPosition = safeget(handles, 'ShowFieldPosition', 1);
    ysmout = ysm(ShowFieldPosition, :)';
    yout = y(ShowFieldPosition, :)';
    
    if ~isempty(areafit)
      BL = mean(y(areafit,:),1)';
      ysmout = ysmout - BL;
      yout = yout - BL;
    end
    
    hand.out = {};
    if handles.cbShowSrc.Value
      hand.out{end+1}.ax = ax;
      hand.out{end}.ax.x = ax.y;
      hand.out{end}.ax.y = ax.x(ShowFieldPosition);
      hand.out{end}.ax.ylabel = ax.xlabel;
      hand.out{end}.y = yout;
      hand.out{end}.ax.Color = [1, 0, 0]; % always red
      hand.out{end}.title = 'original';
    end
    
    hand.out{end+1}.ax = ax;
    hand.out{end}.ax.x = ax.y;
    hand.out{end}.ax.xlabel = ax.ylabel;
    hand.out{end}.ax.y = ax.x(ShowFieldPosition);
    hand.out{end}.ax.ylabel = ax.xlabel;
    hand.out{end}.y = ysmout;
    hand.out{end}.title = 'time trace';
    
    hand.out{end+1}.ax = ax;
    hand.out{end}.ax.x = ax.y(SIG);
    hand.out{end}.ax.y = ax.x(ShowFieldPosition);
    hand.out{end}.y = ysmout(SIG);
    hand.out{end}.ax.xlabel = ax.ylabel;
    hand.out{end}.ax.Color = [0, 1, 0]; % always green
    hand.out{end}.title = 'signal';
    
    if numel(DARK) > 0
      hand.out{end+1}.ax = ax;
      hand.out{end}.ax.x = ax.y(DARK);
      hand.out{end}.ax.y = ax.x(ShowFieldPosition);
      hand.out{end}.ax.xlabel = ax.ylabel;
      hand.out{end}.y = yout(DARK);
      hand.out{end}.ax.Color = [0, 0, 0];
      hand.out{end}.title = 'dark';
    end
  end
end

guidata(handles.hh, hand);
[~,name] = fileparts(get(handles.hh, 'FileName'));
eval([name '(''SetDataSource'', 2, hand)']);

%--------------------------------------------------------------------
function hand = GetHandle(handles)
hand = guidata(handles.hh);

%--------------------------------------------------------------------
function [ax,y,areafit] = GetRawData(handles)
hand = guidata(handles.hh);

ax = hand.src.ax;
y = hand.src.y;

if get(handles.pbTrans,'Value')==1 
    y = y.'; 
    yy = ax.y;
    ax.y = ax.x;
    ax.x = yy;
    ll = ax.xlabel;
    ax.xlabel = ax.ylabel;
    ax.ylabel = ll;
end

if isempty(handles.AreaFit)
  handles.AreaFit = false(size(y,1),1);
  handles.AreaFit(1:10)=true;
  handles.AreaFit(end-10:end)=true;
  guidata(handles.Transient, handles);
end
if size(handles.AreaFit,1) > size(y,1)
  handles.AreaFit = handles.AreaFit(1:size(y,1));
elseif size(handles.AreaFit,1) < size(y,1)
  handles.AreaFit(size(y,1)) = true;
end
areafit = handles.AreaFit;

%--------------------------------------------------------------------
function pbAddRange_Callback(hObject, ~, handles)
hand = guidata(handles.hh);
set(0, 'currentfigure', hand.MainFigure);
disp('Select the beginning and the end of the range');
pos = ginput(2); pos1 = sort(pos(:,1));

[ax,~,areafit] = GetRawData(handles);
areafit(ax.x > pos1(1) & ax.x < pos1(2)) = true;
handles.AreaFit = areafit;
guidata(hObject, handles);
DrawAll(handles);

%--------------------------------------------------------------------
function pbRemoveRange_Callback(hObject, ~, handles)
hand = guidata(handles.hh);
set(0, 'currentfigure', hand.MainFigure);
disp('Select the beginning and the end of the range');
pos = ginput(2); pos1 = sort(pos(:,1));

[ax,~,areafit] = GetRawData(handles);
areafit(ax.x > pos1(1) & ax.x < pos1(2)) = false;
handles.AreaFit = areafit;
guidata(hObject, handles);
DrawAll(handles);

%--------------------------------------------------------------------
function tbFno_Callback(hObject, ~, handles)
allhh = [handles.tbFno, handles.tbF0, handles.tbF1, handles.tbF2, handles.tbF3];
set(allhh, 'Value', 0);
set(hObject, 'Value', 1)

DrawAll(handles);

%--------------------------------------------------------------------
function pbTimeField_Callback(hObject, ~, handles)
handles.TimeField = handles.TimeField+1;
if handles.TimeField > 1, handles.TimeField = 0; end
caps = {'Switch to Field', 'Switch to Time'};
handles.pbTimeField.String = caps{handles.TimeField+1};
caps = {'Time Domain', 'Field Domain'};
% handles.figure1;
guidata(hObject, handles);
DrawAll(handles);

val = {'off','on'};
handles.pbMarkField.Enable = val{handles.TimeField+1};

%--------------------------------------------------------------------
function tbS0_Callback(hObject, ~, handles)
allhh = [handles.tbS0, handles.tbS1, handles.tbS2, handles.tbS3];
set(allhh, 'Value', 0);
set(hObject, 'Value', 1)

DrawAll(handles);

%--------------------------------------------------------------------
function cbShowFit_Callback(~, ~, handles)
DrawAll(handles);

%--------------------------------------------------------------------
function pbSelectSignal_Callback(hObject, ~, handles)
if handles.TimeField == 0
  hand = guidata(handles.hh);
  set(0, 'currentfigure', hand.MainFigure);
  disp('Select the beginning and the end of the range');
  pos = ginput(2); pos1 = sort(pos(:,1));
  
  [ax,~] = GetRawData(handles);
  [~,idx1] = min(abs(ax.y - pos1(1)));
  [~,idx2] = min(abs(ax.y - pos1(2)));
  
  handles.eSstart.String = num2str(idx1);
  handles.eSend.String = num2str(idx2);
  
  DrawAll(handles);
end
%--------------------------------------------------------------------
function pbSelectDark_Callback(hObject, ~, handles)
if handles.TimeField == 0
  hand = guidata(handles.hh);
  set(0, 'currentfigure', hand.MainFigure);
  disp('Select the beginning and the end of the range');
  pos = ginput(2); pos1 = sort(pos(:,1));
  
  [ax,~] = GetRawData(handles);
  [~,idx1] = min(abs(ax.y - pos1(1)));
  [~,idx2] = min(abs(ax.y - pos1(2)));
  
  handles.eBLstart.String = num2str(idx1);
  handles.eBLend.String = num2str(idx2);

  DrawAll(handles);
end
%--------------------------------------------------------------------
function Transient_SizeChangedFcn(~, ~, handles)
standard_size = [0.144, 0.528, 0.18, 0.231];
fig_size = get(handles.Transient, 'Position');
scaling_factor = fig_size(4) / standard_size(4);
DefaultFont = handles.DefaultFont;
if isfield(handles, 'hh')
   hand = guidata(handles.hh);
   DefaultFont = safeget(hand, 'DefaultFont', 8);
end
font_size = DefaultFont + (scaling_factor-1) * (DefaultFont-2);
kazan_size_change(handles.Transient, font_size)

%--------------------------------------------------------------------
function pbMarkField_Callback(hObject, ~, handles)
hand = guidata(handles.hh);
set(0, 'currentfigure', hand.MainFigure);
disp('Select the field position to show trace');
pos = ginput(1);
[ax] = GetRawData(handles);
[~, minidx] = min(abs(ax.x - pos(1)));

handles.ShowFieldPosition = minidx;
guidata(handles.Transient, handles);

%--------------------------------------------------------------------
function tbShowAllTraces_Callback(hObject, eventdata, handles)
DrawAll(handles)
