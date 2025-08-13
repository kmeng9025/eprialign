
function varargout = description(varargin)
% DESCRIPTION Application M-file for description.fig
%    FIG = DESCRIPTION launch description GUI.
%    DESCRIPTION('callback_name', ...) invoke the named callback.

% KAZAN dataviewer with plugins By Boris Epel & Alexey Silakov
% MPI of Bioinorganic Chemistry, Muelhaim an der Ruhr, 2003
% Free for non-commercial use. Use the program at your own risk.
% The authors retains all rights.
% Contact: epel@mpi-muelheim.mpg.de

% Last Modified by GUIDE v2.0 16-Dec-2003 16:46:47

if nargin == 0  % LAUNCH GUI
  token = 'DESCRIPTION_open';
  oldfig = getappdata(0, token);
  oldfig = oldfig(ishandle(oldfig));
  
  [fpath] = fileparts(which('kazan'));
  inifilename = [fpath '\kazan.ini'];
  ini = inimanage(inifilename);
  opc = 2;
  try opc = ini.KazanViewer.MultyKazan; end
  if (opc == 1 || opc == 2) && ~isempty(oldfig), set(oldfig(1),'Visible','on'); varargout{1}=oldfig(1); return; end
  
  fig = openfig(mfilename,'new');
  setappdata(0, token, [oldfig, fig]);
  
  % Generate a structure of handles to pass to callbacks, and store it.
  handles = guihandles(fig);
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
%| 'Description_CloseRequestFcn', 'axis1_ButtondownFcn'.
%|
%| H is the callback object's handle (obtained using GCBO).
%|
%| EVENTDATA is empty, but reserved for future use.
%|
%| HANDLES is a structure containing handles of components in GUI using
%| tags as fieldnames, e.g. handles.Description, handles.slider2. This
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
function Description_DeleteFcn(~, ~, handles)
[~,name] = fileparts(get(handles.hh, 'FileName'));
eval([name '(''DeletePlugins'', handles.Description, handles.hh)']);

% --------------------------------------------------------------------
function varargout = lbDescription_Callback(~, ~, handles, varargin)

% --------------------------------------------------------------------
function FileUpdate(varargin)
handles = varargin{2};
hand = guidata(varargin{2}.hh);
if isfield(hand.src, 'dsc')
  names = fieldnames(hand.src.dsc);
  str = {};
  for k = 1:size(names)
    try
      val = getfield(hand.src.dsc, names{k});
      if isstruct(val)
        s_names = fieldnames(val);
        for jj = 1:size(s_names)
          s_val = getfield(val, s_names{jj});
          str{end+1}=sprintf('%-16s   ',[names{k} '-' s_names{jj}],s_val);
        end
      else
        str{end+1}=sprintf('%-16s   ',[names{k} ':'],val  );
      end
    catch
    end
  end
  set(handles.lbDescription, 'String', str, 'Value', 1);
end

% --------------------------------------------------------------------
function SelectionUpdate(hObject, par, handles)

% --------------------------------------------------------------------
function varargout = pbSeq_Callback(h, ~, handles, varargin)
hand = guidata(handles.hh);
channelnames = {'1','2','TWT','4','detect',...
  '+X','7','8','9','10',...
  '11','12','13','14','15',...
  '16','17','10','19','RFGate',...
  '21','22','23','24','25',...
  '26','27','28','29','RF1',...
  '31','32','33','34','35'};
istep = 1;
if isfield(hand.src, 'dsc')
  if isfield(hand.src.dsc, 'Psd1')
    figure; % (10); clf
    hold on
    val = getsequencestring(hand.src.dsc, 5); % detection channel
    maxval = max(val(:,1)+val(:,2))+50;
    
    for ii=1:36
      val = getsequencestring(hand.src.dsc, ii);
      if any(val(:))
        last_point = 0;
        for j=1:size(val, 1)
          if val(j, 1) > 0 && val(j, 2) > 0
            plot([last_point val(j, 1)], istep+[0 0]);       % _
            plot(val(j,1)*[1,1], istep+[0 .5]);              %  up
            plot(val(j,1)+[0, val(j,2)], istep+[0.5 0.5]);   %   -
            plot((val(j,1)+val(j,2))*[1,1], istep+[0.5 0]);  %    dn
            text(val(j,1),istep+0.7,[num2str(val(j,2)),'ns'])
            text(10,istep+0.3,channelnames{ii})
            last_point = val(j,1)+val(j,2);
          end
        end
        if last_point > 0
          plot([last_point maxval], istep+[0 0]);
          %         text(20, -2*i+.3, num2str(i))
        end
        istep = istep + 1;
      end
    end
    xlabel('Sequence, ns');
    ylabel('Channels');
    axis tight
  elseif isfield(hand.src.dsc, 'XPD1')
    figure; hold on
    for ii=1:16
      if isfield(hand.src.dsc, ['XPD',num2str(ii)])
        start = 0;
        val = getfield(hand.src.dsc, ['XPD' num2str(ii)]);
        val1 = str2num(val);
        if length(val1)>4
          ppulse=floor((length(val1)+3)/4)-1;
          val1(end+1:end+4) = 0;
          for j=1:ppulse
            start1 = val1(1+j*4)*8;
            plot([start,start1], ii+[0 0]);
            text(start+10,ii+0.2,[num2str(start1-start),'ns'])
            start = start1;
            plot([start,start], ii+[0 .5]);
            start1 = 8*(val1(2+j*4)-1)+start;
            plot([start,start1], ii+[0.5 0.5]);
            text(start,ii+0.7,[num2str(start1-start),'ns'])
            plot([start1 start1], ii+[0.5 0]);
            start=start1;
          end
        end
      end
    end
    xlabel('Sequence, ns');
    ylabel('Channels');
    title(hand.src.filename);
    axis tight
  end
end

function val = getsequencestring(dsc, n)
str = safeget(dsc, ['Psd', num2str(n)], '');

pos = strfind(str, '}');
if pos ~= 0
  str = str(pos+1 : end);
end

val = str2num(str);

val = reshape(val, length(val)/4, 4);
