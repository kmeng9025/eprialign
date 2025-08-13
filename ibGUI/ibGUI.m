function varargout = ibGUI(varargin)
% Ortho-slice viewer with additional features
% Usage:
%   ibGUI - open GUI
%   ibGUI(3D_or_4D_matrix) - browse one matrix
%   ibGUI(a_struct) - browse a set of images, see example below
%         option parameters are optional 
%         the_roi = struct('isROI', true, 'ROI', 32);
%         the_options = struct('pO2', struct('minmax', [0,50]));    
%         a_struct = struct('pO2', 3D_matrix, 'RAW', 4D_matrix, 'Mask',
%                            image_mask, ..., 'options', the_options);
%         ibGUI(a_struct)
%   Fields:
%     Size
%     

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, JANUARY 2014
% Contact: epri.uchicago.edu

% Last Modified by GUIDE v2.5 11-Jan-2023 12:18:32

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @ibGUI_OpeningFcn, ...
  'gui_OutputFcn',  @ibGUI_OutputFcn, ...
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


% --- Executes just before ibGUI is made visible.
function ibGUI_OpeningFcn(hObject, eventdata, handles, varargin)

% handles.mWinImageCompare = uimenu(handles.mWindow, 'Label','ImageCompareGUI', 'Tag', 'mWinImageCompare', ...
%   'Callback', 'ImageCompareGUI');
handles.mWinProcessAPF = uimenu(handles.mWindow, 'Label','ProcessAPFGUI', 'Tag', 'mWinProcessAPF', ...
  'Callback', 'ProcessAPFGUI');
handles.mWinPulseRecon = uimenu(handles.mWindow, 'Label','PulseReconGUI', 'Tag', 'mWinPulseRecon', ...
  'Callback', 'PulseReconGUI');
% handles.mWinPulseFit = uimenu(handles.mWindow, 'Label','PulseFitGUI', 'Tag', 'mWinPulseFit', ...
%   'Callback', 'PulseFitGUI');
% handles.mWinCWRecon = uimenu(handles.mWindow, 'Label','recon_gui', 'Tag', 'mWinCWRecon', ...
%   'Callback', 'recon_gui');
% handles.mWinCWFit = uimenu(handles.mWindow, 'Label','FitGUI', 'Tag', 'mWinCWFit', ...
%   'Callback', 'FitGUI');
% handles.mWinMaskEdit = uimenu(handles.mWindow, 'Label','MaskEditGUI', 'Tag', 'mWinMaskEdit', ...
%   'Callback', 'MaskEditGUI');

% copy settings from the previous window
if nargin == 4 && isfield(varargin{1}, 'transfer')
  transfer = varargin{1};
  
  
  flds = {'ini','output','options','plot4','ROIrad','hit_ax','ax123info','ax4info','TumorEllipsoid','REFxyz','NCxyz','tracks','A',...
    'auto_colors','MaskToolBoxPos','projection','Toolbars','op_mode','op_mode_busy','StartFunction','EndFunction','ImageDim','is2D','ImageSize',...
    'RawImage', 'export', 'Timeline'};
  
  setappdata(handles.figure1, 'Images', transfer.Images);
  setappdata(handles.figure1, 'Masks', transfer.Masks);
  setappdata(handles.figure1, 'MaskUndo', []);
  for ii=1:length(flds)
    if isfield(transfer, flds{ii})
      handles.(flds{ii}) = transfer.(flds{ii});
    end
  end
  
  str = {};
  for ii=1:length(transfer.Images)
    str{end+1} = ['Slices: ',transfer.Images{ii}.Type];
  end
  set(handles.pmImageType, 'String', str, 'Value', 1);

  % Update handles structure
  guidata(hObject, handles);
  set(handles.pMaskControl, 'Visible', 'off');
  UpdateMaskList(handles);
  AfterProjectionChanged(handles);
else
  handles.ini = inimanage(epr_GetIniPath('ImageBrowserGUI'));
  
  % Initialize Java class
  % dndcontrol.initJava();
  
  % jFrame = getjframe(handles.figure1);
  % % Create Java Swing JScrollPane
  % jScrollPane = javaObjectEDT('javax.swing.JScrollPane', jTextArea);
  % jScrollPane.setVerticalScrollBarPolicy(jScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
  %
  % % Add Scrollpane to figure
  % [~,hContainer] = javacomponent(jScrollPane,[],hFig);
  % set(hContainer,'Units','normalized','Position',[0 0 1 1]);
  %
  % % Create dndcontrol for the JTextArea object
  % dndobj = dndcontrol(jTextArea);
  %
  % % Set Drop callback functions
  % dndobj.DropFileFcn = @demoDropFcn;
  % dndobj.DropStringFcn = @demoDropFcn;
  
  
  handles.output.hh = hObject;
  handles.output.result = 'OK';
  handles.options.PathName = '';
  handles.options.FileName = '';
  handles.options = struct('Max', 1.0, 'Min', 1.0, 'Reference', 'MinMax', ...
    'ConstDialog', false, 'Erosion', 0, 'ErosionROI', 0, ...
    'ExternalMask', false, 'ExternalMaskType', 'AsIs');
  handles = SetLabel(handles, 'EPRI');
  handles.plot4 = 2;
  handles.ROIrad = 0;
  handles.hit_ax = 1;
  handles.ax123info.idx = [1,2,3];
  handles.ax4info = [];
  
  handles.TumorEllipsoid.expInfo.magnetType = 0;
  handles.TumorEllipsoid.expInfo.resonType = '';
  handles.TumorEllipsoid.expInfo.rGap = [0, 0, 0];
  handles.TumorEllipsoid.Mask = [];
  handles.TumorEllipsoid.Resonator = [];
  
  handles.REFxyz = [0,0,0];
  handles.NCxyz  = [0,0,0];
  
  handles.tracks = {};
  handles.A = eye(4);
  handles.auto_colors = {'k','m','k','c', 'r', 'g'};
  % old color orderMM changed
  % handles.auto_colors = {'k', 'm', 'r', 'g', 'k', 'm', 'r', 'g', 'm', 'k', 'r', 'g', 'm', 'k', 'r', 'g'};
  handles.MaskToolBoxPos = 4;
  
  setappdata(handles.figure1, 'Images', {});
  setappdata(handles.figure1, 'Masks', {});
  handles.projection = [];
  
  guidata(hObject, handles);
  
  % toolbars
  handles.Toolbars.hMaskToolbar = [];
  handles.Toolbars.hMaskToolbar2 = [];
  set(handles.pMaskControl, 'Visible', 'off');
  
  handles.op_mode = [];
  handles.op_mode_busy = false;
  
  % options supplied
  theFileName = '';
  isOptionSupplied = false;
  if nargin > 4 && ~ischar(varargin{2})
    B = varargin{2};
    for fn = fieldnames(B)'
      handles.options.(fn{1}) = B.(fn{1});
    end
    isOptionSupplied = true;
  else
    if nargin > 4 && ischar(varargin{2})
      theFileName = varargin{2};
    end
  end
  
  handles.options.PathName = '';
  handles.options.CWload = {'CC', 'pO2', 'ERROR', 'RAW', 'SNR', 'LW', 'PHASE', 'XOVER','ERROR_STD', 'ERROR_O2'};
  handles.options.CWidx  = [1,2,3,4];
  handles.options.ESEload = {'CC', 'pO2', 'ERROR', 'RAW', 'T2', 'T1', 'LW','ERROR_T2','ERROR_T1','ERROR_STD', 'ERROR_O2'};
  handles.options.ESEidx  = [1,2];
  
  % Initialize predefined image types
  field_def = {'Amp', 'pO2', 'Error', 'Raw', 'PHASE', 'XOVER', 'T2', 'T1', 'LW', 'Error_T2', 'Error_T1', 'Error_STD', 'Error_O2'};
  default_range = {[0,1],[0, 100],[],[],[-180,180],[],[0,7],[0,8],[],[],[],[], [0,100]};
  field_dsc = {'Amplitude', 'pO2', 'Norm fit error', 'Raw image', ...
    'Phase (CW)', 'Crossover(CW)', 'T2 relaxation', 'T1 relaxation', 'Lorentzian linewidth', 'T2 norm error', 'T1 norm error', 'Fit error', 'O2 error'};
  field_unit = {'mM', 'torr', 'au', 'au', ...
    'rad', 'mG', 'us', 'us', 'mG', 'au', 'au', 'au', 'torr'};
  for ii=1:length(field_def)
    handles.options.settings.(field_def{ii}).default_range = default_range{ii};
    handles.options.settings.(field_def{ii}).field_dsc = field_dsc{ii};
    handles.options.settings.(field_def{ii}).field_unit = field_unit{ii};
  end
  
  % Mask related options
  handles.options.isPreserveDataMask = safeget(handles.options, 'isPreserveDataMask', true);
  handles.options.isXRayView = safeget(handles.options, 'isXRayView', false);
  handles.options.isContoursRestrict = safeget(handles.options, 'isContoursRestrict', false);
  handles.options.isMaskRaw = safeget(handles.options, 'isMaskRaw', false);
  
  opt = {'off', 'on'};
  set(handles.mContoursPreserveDataMAsk, 'Checked', opt{handles.options.isPreserveDataMask+1});
  set(handles.mContoursXRay, 'Checked', opt{handles.options.isXRayView+1});
  set(handles.mContoursRestrict, 'Checked', opt{handles.options.isContoursRestrict+1});
  set(handles.mContoursMaskRaw, 'Checked', opt{handles.options.isMaskRaw+1});
  
  guidata(hObject, handles);
  
  % Image supplied
  handles.StartFunction = [];
  handles.EndFunction = [];
  if nargin > 3 && ~isempty(varargin{1})
    if ischar(varargin{1})
      handles = Data_Load(varargin{1}, handles, true, true);
    elseif isstruct(varargin{1})
      handles.StartFunction = safeget(varargin{1}, 'StartFunction', []);
      handles.EndFunction = safeget(varargin{1}, 'EndFunction', []);
      handles.ParentHandle = safeget(varargin{1}, 'ParentHandle', []);
      handles = Data_Load(varargin{1}, handles, false, ~isOptionSupplied);
      if ~isempty(handles.StartFunction)
        handles = guidata(handles.figure1);
        handles = handles.StartFunction(handles.ParentHandle, handles);
      end
      if ~isempty(theFileName)
        SetImageName(handles, theFileName);
      end
    else
      a.Raw = varargin{1};
      handles = Data_Load(a, handles, false, true);
    end
    %   Mask_Check(handles.figure1, handles.ImageDim);
    UpdateMaskList(handles);
    AfterProjectionChanged(handles);
  else
    % Update handles structure
    guidata(hObject, handles);
  end
end

% --------------------------------------------------------------------
function figure1_CloseRequestFcn(hObject, eventdata, handles)
try
  if isfield(handles, 'ini')
    if ~isempty(handles.EndFunction)
      handles.EndFunction(handles.ParentHandle, handles.figure1);
    end
    inimanage(epr_GetIniPath('ibGUI'), handles.ini);
  end
catch err
  error_catcher('figure1_CloseRequestFcn', err)
end
delete(hObject);

% --------------------------------------------------------------------
function varargout = ibGUI_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;

% --------------------------------------------------------------------
function mFileCloseOK_Callback(hObject, eventdata, handles)
handles.output.result = 'OK';
guidata(hObject, handles);
close(handles.figure1);

% --------------------------------------------------------------------
function mFileCloseCancel_Callback(hObject, eventdata, handles)
handles.output.result = 'Cancel';
guidata(hObject, handles);
close(handles.figure1);

% --------------------------------------------------------------------
function figure1_ResizeFcn(hObject, eventdata, handles)
% Obtain size of the window
fig_size = get(handles.figure1, 'Position');

% Main window layout
mborder_t = 20; mborder_b = 10; mborder_l = 14; mborder_r = 18;
h_panel = 150;
h_tools = fig_size(3) - h_panel;

% Axis layout
h_label_size = 15; h_axis_space = 15;
v_label_size = 10; v_axis_space = 10;

haxis_size = (h_tools - mborder_r - mborder_l - h_axis_space) / 2;
vaxis_size = (fig_size(4) - mborder_t - mborder_b - v_axis_space) / 2;

pos1 = [mborder_l+h_label_size, mborder_b+v_label_size, haxis_size-h_label_size, vaxis_size-v_label_size];
set(handles.axes_3, 'Position', pos1)
set(handles.axes_1, 'Position', pos1 + [0,vaxis_size+v_axis_space+v_label_size,0,0])
set(handles.axes_2, 'Position', pos1 + [haxis_size+h_axis_space,vaxis_size+v_axis_space+v_label_size,0,0])

pos2 = pos1 + [haxis_size+h_axis_space,v_label_size*3,0,-3*v_label_size];
set(handles.axesAdt, 'Position', pos2)

% Mask panel
enlarge = [-16, 0, 20, 10];
switch safeget(handles, 'MaskToolBoxPos', 4)
  case 1, pos4 = pos1 + [0,vaxis_size+v_axis_space+v_label_size,0,0]+enlarge;
  case 2, pos4 = pos1 + [haxis_size+h_axis_space,vaxis_size+v_axis_space+v_label_size,0,0]+enlarge;
  case 3, pos4 = pos1+enlarge;
  case 4, pos4 = pos1 + [haxis_size+h_axis_space,0,0,0]+enlarge;
end
btn_size = 45;
set(handles.pMaskControl, 'Position', pos4);
set(handles.pmNewMask, 'Position', [5, pos4(4)-45, pos4(3)-18-btn_size, 25]);
set(handles.pbAddMask, 'Position', [pos4(3)-btn_size-9, pos4(4)-45, btn_size, 25]);
set(handles.lbMaskList, 'Position', [5, 5, pos2(3)-18-btn_size, pos4(4) - 55]);
set(handles.pbRemoveMask, 'Position', [pos4(3)-btn_size-9, pos4(4)-70, btn_size, 25]);
set(handles.pbShowMask, 'Position', [pos4(3)-btn_size-9, pos4(4)-99, btn_size, 25]);
set(handles.pbEditMask, 'Position', [pos4(3)-btn_size-9, pos4(4)-123, btn_size, 25]);

set(handles.pbPos1, 'Position', [pos4(3)-btn_size-9, pos4(4)-150, btn_size/2, 20]);
set(handles.pbPos2, 'Position', [pos4(3)-btn_size-9+btn_size/2, pos4(4)-150, btn_size/2, 20]);
set(handles.pbPos3, 'Position', [pos4(3)-btn_size-9, pos4(4)-150-22, btn_size/2, 20]);
set(handles.pbPos4, 'Position', [pos4(3)-btn_size-9+btn_size/2, pos4(4)-150-22, btn_size/2, 20]);

% Tools layout
v_tools = fig_size(4)-10;
set(handles.ToolPanel, 'Position', [h_tools+2, 5, h_panel-10, v_tools])

tborder_t = 20; tborder_b = 10; tborder_l = 4; tborder_r = 18;
esize_v = 21; esize_h = 58; slsize_h = 40; tsize_h = 20; espace_v = 2; espace_h = 4;

set(handles.cbConnect12321, 'Position', [tborder_l+70, v_tools-30, 60, 17])

box_offset = [0,-esize_v-espace_v, 0, 0];
% Edit boxes
top_slice_controls = v_tools-(tborder_t+esize_v+11);
pos1 = [tborder_l+espace_h+tsize_h, top_slice_controls, esize_h, esize_v];
set(handles.editDim1, 'Position', pos1)
set(handles.editDim2, 'Position', pos1+box_offset)
set(handles.editDim3, 'Position', pos1+2*box_offset)
set(handles.editDim4, 'Position', pos1+3*box_offset);

% Text boxes
pos1 = [tborder_l, top_slice_controls-3, tsize_h, esize_v];
set(handles.text_i, 'Position', pos1)
set(handles.text_j, 'Position', pos1+box_offset)
set(handles.text_k, 'Position', pos1+2*box_offset)
set(handles.text_l, 'Position', pos1+3*box_offset)

% Slider boxes
pos1 = [tborder_l+2*espace_h+esize_h+tsize_h, top_slice_controls, slsize_h, esize_v];
set(handles.sliderDim1, 'Position', pos1)
set(handles.sliderDim2, 'Position', pos1+box_offset)
set(handles.sliderDim3, 'Position', pos1+2*box_offset)
set(handles.sliderDim4, 'Position', pos1+3*box_offset)

% Mode selector controls and stat
msize_v = top_slice_controls - (3*esize_v+4*espace_v);
pos1 = [tborder_l, msize_v - esize_v, h_panel-tborder_l-tborder_r, esize_v];
set(handles.pmImageType, 'Position', pos1)
set(handles.pm4thplot, 'Position', pos1 + box_offset)
set(handles.textStat, 'Position', pos1 + 2*box_offset)
set(handles.textStat1, 'Position', pos1 + 2.8*box_offset)

lwsize_v = 60;
pos1 = [tborder_l, msize_v - 85 - lwsize_v, h_panel-tborder_l-tborder_r, lwsize_v];
set(handles.eLogWindow, 'Position', pos1)

% Scale controls
scsize_v = msize_v - 85 - lwsize_v; scspace_v = 10;
pos1 = [tborder_l, tborder_b, 25, scsize_v - tborder_b - scspace_v];
set(handles.axScale, 'Position', pos1)

pos1 = [tborder_l+68, scsize_v-esize_v-scspace_v, 60, esize_v];
set(handles.eShowMax, 'Position', pos1)
set(handles.cbRange12321, 'Position', pos1+box_offset)
pos1 = [tborder_l+68, tborder_b, 60, esize_v];
set(handles.eShowMin, 'Position', pos1);
pos1 = [tborder_l+50, tborder_b, 18, esize_v];
set(handles.pb0, 'Position', pos1);

bsize_h = (60-4)/2;
pos1 = [tborder_l+68, tborder_b+esize_v+5, bsize_h, 26];
set(handles.pbDefaultRange, 'Position', pos1)
pos1 = [tborder_l+68+bsize_h+4, tborder_b+esize_v+5, bsize_h, 26];
set(handles.pbResetRange, 'Position', pos1)

% --------------------------------------------------------------------
function handles = SetLabel(handles, label_set)
switch label_set
  case 'EPRI'
    set(handles.text_i, 'String', '1:X');
    set(handles.text_j, 'String', '2:Y');
    set(handles.text_k, 'String', '3:Z');
    set(handles.text_l, 'String', '4:B');
    handles.options.axis_label = {'X', 'Y', 'Z', 'B'};
    handles.options.axis_idx   = {1, 2, 3, 4};
    handles.options.plots_label = {'YX', 'ZX', 'YZ', 'B'};
    handles.options.plots_idx   = {[2,1], [3,1], [2,3]};
    handles.options.plots_idx1   = [3, 2, 1];
end

% --------------------------------------------------------------------
function eLogWindow_Callback(hObject, eventdata, handles) 

% --------------------------------------------------------------------
function pmImageType_Callback(hObject, eventdata, handles) 
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
function editDim_Callback(hObject, eventdata, handles) 
val = str2double(get(hObject, 'String'));
idx = handles.options.axis_idx;
switch hObject
  case handles.editDim1, handles.projection(idx{1}) = val;
  case handles.editDim2, handles.projection(idx{2}) = val;
  case handles.editDim3, handles.projection(idx{3}) = val;
  case handles.editDim4, handles.projection(idx{4}) = val;
end
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
function sliderDim_Callback(hObject, eventdata, handles) 
val = get(hObject, 'Value');
set(hObject, 'Value', 0);
idx = handles.options.axis_idx;

switch hObject
  case handles.sliderDim1, handles.projection(idx{1}) = val+handles.projection(idx{1});
  case handles.sliderDim2, handles.projection(idx{2}) = val+handles.projection(idx{2});
  case handles.sliderDim3, handles.projection(idx{3}) = val+handles.projection(idx{3});
  case handles.sliderDim4, handles.projection(idx{4}) = val+handles.projection(idx{4});
end
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
function figure1_WindowButtonDownFcn(hObject, eventdata, handles) 
if handles.op_mode_busy, return; end
if ~isfield(handles, 'ImageSize'), return; end

cMask = classMaskStorage(handles.figure1, 'Masks', handles.options);
[hit_ax, ijk] = GetMousePosition(handles);

if ~isempty(hit_ax)
  handles.hit_ax = hit_ax;
  ax = [handles.axes_1, handles.axes_2, handles.axes_3];
  %   disp(handles.op_mode)
  if strcmp(handles.op_mode, 'AddPoly') || strcmp(handles.op_mode, 'ErasePoly') || ...
      strcmp(handles.op_mode, 'AddFreeHand') || strcmp(handles.op_mode, 'EraseFreeHand')
    handles.op_mode_busy = true;
    guidata(handles.figure1, handles);
    try
      if strcmp(handles.op_mode, 'AddPoly') || strcmp(handles.op_mode, 'ErasePoly')
        h = impoly(ax(hit_ax), []);
      elseif strcmp(handles.op_mode, 'AddFreeHand') || strcmp(handles.op_mode, 'EraseFreeHand')
        h = imfreehand(ax(hit_ax));
      end
      api = iptgetapi(h);
      handles.op_mode_busy = false;
      
      ijk = XYZtoIJK(handles, api.getPosition(), hit_ax);
      
      sz = handles.ImageDim;
      if length(sz) > 3, sz = sz(1:3); end
      switch hit_ax
        case 1, Mask = poly2mask(ijk(:,2), ijk(:,1), sz(2), sz(1)); Slice = handles.projection(3);
        case 2, Mask = poly2mask(ijk(:,1), ijk(:,3), sz(3), sz(1))'; Slice = handles.projection(2);
        case 3, Mask = poly2mask(ijk(:,3), ijk(:,2), sz(2), sz(3)); Slice = handles.projection(1);
      end
      mask_idx = get(handles.lbMaskList, 'Value');
      if strcmp(handles.op_mode, 'AddPoly') || strcmp(handles.op_mode, 'AddFreeHand')
        Mask_ApplySlice(handles.figure1, Mask, sz, mask_idx, hit_ax, Slice, 'add', handles.options)
      elseif  strcmp(handles.op_mode, 'ErasePoly') || strcmp(handles.op_mode, 'EraseFreeHand')
        Mask_ApplySlice(handles.figure1, Mask, sz, mask_idx, hit_ax, Slice, 'erase', handles.options)
      end
    catch err
      disp(err)
    end
    delete(h);
    handles.op_mode = [];
    handles.op_mode_busy = false;
    AfterProjectionChanged(handles);
  elseif strcmp(handles.op_mode, 'SelectByThreshold')
    try
      image_number = get(handles.pmImageType, 'Value');
      [Data, slice_number] = Data_GetSlice(handles.figure1, image_number, hit_ax, handles.projection);
      answer=inputdlg({'Lower threshold'; 'Higher threshold'},...
        'Mask parameters',1, {num2str(min(Data(:))); num2str(max(Data(:)))});
      min_thres = str2double(answer{1});
      max_thres = str2double(answer{2});
      mask_idx = get(handles.lbMaskList, 'Value');
      Mask_ApplySlice(handles.figure1, Data >= min_thres & Data <= max_thres, handles.ImageDim, mask_idx, hit_ax, slice_number, 'add', handles.options)
    catch err
      error_catcher('WindowButtonDownFcn:SelectByThreshold', err);
    end
    handles.op_mode = [];
    handles.op_mode_busy = false;
    AfterProjectionChanged(handles);
  elseif strcmp(handles.op_mode, 'Otsu')
    try
      image_number = get(handles.pmImageType, 'Value');
      
      [Data, slice_number] = Data_GetSlice(handles.figure1, image_number, hit_ax, handles.projection);
      
      answer=inputdlg({'Number of thresholds'; 'Use range'},...
        'Otsu algorithm parameters',1, {'4';'4'});
      if ~isempty(answer)
        n_threshold = fix(str2double(answer{1}))-1;
        use_threshold = fix(str2double(answer{2}));
        mask_idx = get(handles.lbMaskList, 'Value');
        Data = Data / max(Data(:));
        
        Levels = multithresh(Data,n_threshold);
        NewMask = imquantize(Data,Levels);
        %      ibGUI(NewMask);
        Mask_ApplySlice(handles.figure1, NewMask == use_threshold, handles.ImageDim, mask_idx, hit_ax, slice_number, 'replace', handles.options)
      end
    catch err
      error_catcher('WindowButtonDownFcn:Otsu', err);
    end
    handles.op_mode = [];
    handles.op_mode_busy = false;
    AfterProjectionChanged(handles);
  elseif strcmp(handles.op_mode, 'EraseAll') || strcmp(handles.op_mode, 'SetAll')
    try
      image_number = get(handles.pmImageType, 'Value');
      [Data, slice_number] = Data_GetSlice(handles.figure1, image_number, hit_ax, handles.projection);
      mask_idx = get(handles.lbMaskList, 'Value');
      if strcmp(handles.op_mode, 'EraseAll')
        Mask_ApplySlice(handles.figure1, true(size(Data)), handles.ImageDim, mask_idx, hit_ax, slice_number, 'erase', handles.options)
      else
        Mask_ApplySlice(handles.figure1, true(size(Data)), handles.ImageDim, mask_idx, hit_ax, slice_number, 'add', handles.options)
      end
    catch err
      error_catcher('WindowButtonDownFcn:EraseAll/SetAll', err);
    end
    handles.op_mode = [];
    handles.op_mode_busy = false;
    AfterProjectionChanged(handles);
  elseif strcmp(handles.op_mode, 'Interpolate')
    try
      mask_idx = get(handles.lbMaskList, 'Value');
      InterpolatedMask = Data_ToZ(cMask.Get(mask_idx), hit_ax);

      % determine the scope of interpolation
      nvox = squeeze(sum(sum(InterpolatedMask, 1), 2));
      slices = nvox ~= 0;
      sz = size(InterpolatedMask);
      max_idx = find(slices,1,'last');
      min_idx = find(slices,1,'first');
      if isempty(min_idx) || isempty(max_idx), return; end
      applied_slices = min_idx:max_idx;
      applied_slices = applied_slices(~slices(applied_slices));
      if isempty(applied_slices), return; end

      % calculate number of voxels expected
      nvox_exp = interp1(find(slices), nvox(slices), applied_slices);

      % build distance volume in available planes
      interp_mask = InterpolatedMask(:,:, slices);
      sz_interp = size(interp_mask);

      volout=ones(sz_interp)*max([sz_interp(1) sz_interp(2)])/2;
      for ns=1:sz_interp(3)
        imsk=squeeze(interp_mask(:,:,ns));
        dout=bwdist(imsk);
        din=bwdist(~imsk);
        volout(:,:,ns)=dout-din;
      end
      
      % interpolate this volume
      [x,y,z]=meshgrid(1:sz(2), 1:sz(1), 1:sz(3));
      new_mask = interp3(x(:,:, slices), y(:,:, slices), z(:,:, slices), volout, ...
        x(:,:, applied_slices), y(:,:, applied_slices), z(:,:, applied_slices));
      
      % Threshold to account for expected voxels
      for ii=1:length(applied_slices)
        cut_level = 1;
        the_slice = new_mask(:,:,ii);
        while numel(find(the_slice(:) < cut_level)) < nvox_exp(ii)
          cut_level=cut_level+1;
        end
        InterpolatedMask(:,:,applied_slices(ii)) = the_slice < cut_level;
      end
      cMask = classMaskStorage(handles.figure1, 'Masks', handles.options);
      cMask.Set(mask_idx, Data_FromZ(InterpolatedMask, hit_ax));
    catch err
      error_catcher('WindowButtonDownFcn:Interpolate', err);
    end
    handles.op_mode = [];
    handles.op_mode_busy = false;
    AfterProjectionChanged(handles);
  else
    if hit_ax == 4
      switch handles.plot4
        case 6
          handles.projection = ijk;
          AfterProjectionChanged(handles);
      end
    else
      handles.projection = ijk;
      AfterProjectionChanged(handles);
    end
  end
end

% --------------------------------------------------------------------
function figure1_WindowButtonMotionFcn(hObject, eventdata, handles) 

if ~isfield(handles, 'ImageSize'), return; end
[hit_ax, ijk, xyz] = GetMousePosition(handles);

if ~isempty(hit_ax) && length(xyz) > 2
  image_number = get(handles.pmImageType, 'Value');
  v = Data_GetXYZ(handles.figure1, image_number, ijk);
  set(handles.textStat, 'String', sprintf('%5f [%i;%i;%i]', v, ijk(1:3)))
  set(handles.textStat1, 'String', sprintf('[%5.2f; %5.2f; %5.2f]', xyz(1:3)))
end

% --------------------------------------------------------------------
function [ijk, xyz] = XYZtoIJK(handles, xy, hit_ax)
% convert 2D xy coordinates of the selected axis into matrix indexes ijk
% and full coordinates xyz

xyztmp = [handles.projection(1:3),1]/handles.A;
nxy = size(xy,1);
xyz = zeros(nxy, 4);
for ii=1:nxy
  xyz(ii,:) = FullProjection(xyztmp, xy(ii,:), hit_ax, Plot_Idx(handles.options.plots_idx, handles.ax123info.idx));
end
ijk = fix(xyz*handles.A + 0.5); ijk(:,4) = handles.projection(4);

% --------------------------------------------------------------------
function hit_ax_idx = GetActiveAxis(handles, CoordXY)
% find the axis hit by mouse
plot_axes = [handles.axes_1, handles.axes_2, handles.axes_3, handles.axesAdt];
plot_Pos = get(plot_axes, 'Position');

hit_ax_idx = [];
for ii=1:length(plot_axes)
  if CoordXY(1)>=plot_Pos{ii}(1)&& CoordXY(1)< sum(plot_Pos{ii}([1, 3])) && ...
      CoordXY(2)>plot_Pos{ii}(2)&& CoordXY(2)< sum(plot_Pos{ii}([2, 4]))
    hit_ax_idx = ii;
    break;
  end
end

% --------------------------------------------------------------------
function [hit_ax, ijk, xyz] = GetMousePosition(handles)
% Calculates cursor position in axis units
% from coordinates received using 'CurrentPoint' of CurrentAxes.


CoordXY = get(handles.figure1, 'CurrentPoint');
hit_ax  = GetActiveAxis(handles, CoordXY);
if isempty(hit_ax), ijk = [0 0 0 0]; xyz = [0 0 0 0]; return; end

plot_axes = [handles.axes_1, handles.axes_2, handles.axes_3, handles.axesAdt];
CXY = get(plot_axes(hit_ax), 'CurrentPoint');
xy  = CXY(1, (1:2));

if handles.is2D && hit_ax ~= 1,  ijk = [0 0 0 0]; xyz = [0 0 0 0]; return; end
  
if hit_ax ~= 4
  [ijk, xyz] = XYZtoIJK(handles, xy, hit_ax);
else
   ijk = [0 0 0 0]; xyz = [0 0 0 0]; 
  switch handles.plot4
    case 6
      A = handles.A;
      track = safeget(handles, 'ax4info', struct('idx1', [], 'idx2', [], 'idx3', [], 'min_max', [0 1]));
      
      x = max(floor(xy(1) + 0.5), 1);
      y =  max(floor((xy(2) - track.min_max(1))/diff(track.min_max)*length(track.idx3) + 0.5), 1);
      if x <= length(track.idx1) && y <= length(track.idx3)
        ijk(1:3) = ...
          ImageInNormalAxis([track.idx1(x), track.idx2(x), track.idx3(y)], track.hit_ax);
        ijk(4) = handles.projection(4);
        xyz = [ijk(1:3),1]*inv(A);
      end
    otherwise
      xyz = xy;
  end
end

% --------------------------------------------------------------------
function plots_idx = Plot_Idx(plots_idx, new_idx)
plots_idx{1} = new_idx(plots_idx{1});
plots_idx{2} = new_idx(plots_idx{2});
plots_idx{3} = new_idx(plots_idx{3});

% --------------------------------------------------------------------
function Projection = FullProjection(Projection, XY, SliceN, plots_idx)
if isempty(Projection), Projection=zeros(1, 4); end
Projection(plots_idx{SliceN}) = XY;

% --------------------------------------------------------------------
function handles = AfterProjectionChanged(handles, arg2)
try
  handles = UpdateProjection(handles);
  
  Draw_Image(handles, handles.projection)
  Draw_ROI_Info(handles);
  if ~exist('arg2', 'var') && get(handles.cbConnect12321, 'Value')
    h = GetOtherGUI(handles);
    
    for ii=1:length(h)
      hh = guidata(h(ii));
      hh.projection = fix(handles.projection); % .*hh.ImageSize./handles.ImageSize
      AfterProjectionChanged(hh, 0);
    end
  end
catch err
  fprintf('ibGUI: Draw_Image error: %s.\n', err.message);
  guidata(handles.figure1, handles);
end

% --------------------------------------------------------------------
function hh = GetOtherGUI(handles)
set(0, 'ShowHiddenHandles', 'on');
h = findobj('Tag', 'cbConnect12321');
set(0, 'ShowHiddenHandles', 'off');
hh = [];
for ii=1:length(h)
  if h(ii) ~= handles.cbConnect12321 && get(h(ii), 'Value')
    hh(end+1) = h(ii);
  end
end
% --------------------------------------------------------------------

function SetImageName(handles, image_name)   
set(handles.figure1, 'Name', ['ibGUI v.2.0: ', image_name])

% --------------------------------------------------------------------
function handles = Data_Load(file_name, handles, is_load, is_stat)
if is_load
  pars.CW  = handles.options.CWload(handles.options.CWidx(:));
  pars.ESE = handles.options.ESEload(handles.options.ESEidx(:));
  try
    % Analyse the extension
    [handles.options.PathName,~,ext] = fileparts(file_name);
    handles.options.FileName = file_name;
    handles.ini.Directories.SourcePath = file_name;
    SetImageName(handles, file_name);
    % MRI files have no extensions
    if isempty(ext), ext = '.IMG'; end
    switch upper(ext)
      case '.MAT'
        LoadedImage = epr_LoadMATFile(file_name, true, pars);
      case '.IMG'
        LoadedImage = epr_LoadBrukerMRI(file_name);
      case '.TDMS'
        LoadedImage = epr_LoadOxyVue(file_name);
    end
  catch err
    error_catcher('Data_Load', err)
    handles.ini.Directories.SourcePath = file_name;
    guidata(handles.figure1, handles);
    return;
  end
else
  LoadedImage = file_name;
  handles.ini.Directories.SourcePath = safeget(LoadedImage, 'ImageName', '');
  if isfield(LoadedImage, 'ImageName')
    SetImageName(handles, LoadedImage.ImageName);   
  end 
end

get_file_info(LoadedImage);

% generate selection list by selecting numerical fields with number of
% dimensions exceeding one
Images = []; im_idx = 1; ax4info = [];
setappdata(handles.figure1, 'MaskUndo', []);
str = {};
fields = fieldnames(LoadedImage);
for ii=1:length(fields)
  if isstruct(LoadedImage.(fields{ii})), continue; end
  if strcmp(fields{ii}, 'Mask'), continue; end
  if strcmp(fields{ii}, 'Masks'), continue; end
  % Turn matrix into double
  if islogical(LoadedImage.(fields{ii}))
      LoadedImage.(fields{ii}) = double(LoadedImage.(fields{ii}));
  end
  if ~isnumeric(LoadedImage.(fields{ii})), continue; end
  sz = size(LoadedImage.(fields{ii}));
  if length(find(sz > 1)) <= 1, continue; end
  
  str{end+1} = ['Slices: ',fields{ii}];
  Images{im_idx}.Image = LoadedImage.(fields{ii});
  Images{im_idx}.Type = fields{ii};
  Images{im_idx}.Dim123Unit = 'cm';
  Images{im_idx}.StartTime = safeget(LoadedImage, 'StartTime', []);
  Images{im_idx}.FinishTime = safeget(LoadedImage, 'FinishTime', []);
  if ndims(Images{im_idx}.Image) == 4
    ax4info.RawImage = im_idx;
  end
  im_idx = im_idx + 1;
end
ax4info.Dim4Type = safeget(LoadedImage, 'Dim4Type', []);
set(handles.pmImageType, 'String', str, 'Value', 1);
setappdata(handles.figure1, 'Images', Images);

% Find image dimensions, create data mask, update options for each image
if ~isfield(LoadedImage, 'Mask'), LoadedImage.Mask = []; end
[handles.ImageDim, handles.options] = Data_Check(handles.figure1, LoadedImage.Mask, handles.options);
handles.is2D = iff(handles.ImageDim(3) == 1, true, false);
if isfield(LoadedImage, 'Masks')
  for ii=1:length(LoadedImage.Masks)
    Mask_Add(handles.figure1, LoadedImage.Masks{ii}.Mask, LoadedImage.Masks{ii}.Name);
  end
end

% change scaling options
if isfield(LoadedImage, 'options')
  for ii=1:length(Images)
    image_name = Images{ii}.Type;
    if isfield(LoadedImage.options, image_name)
      new_options = LoadedImage.options.(image_name);
      if isfield(new_options, 'minmax')
        handles.options.(image_name).minmax = new_options.minmax;
        handles.options.(image_name).scale_minmax = new_options.minmax;
      end
    end
  end
end

% Find image physical size
if handles.is2D
  if isfield(LoadedImage, 'Size')
    ImageSize = safeget(LoadedImage, 'Size', handles.ImageDim(1));
    if numel(ImageSize) < 2, ImageSize(2) = ImageSize(1); end
  else
    ImageSize = handles.ImageDim(1:3);
  end
else
  if isfield(LoadedImage, 'Size')
    ImageSize = safeget(LoadedImage, 'Size', handles.ImageDim(1));
    if numel(ImageSize) < 2, ImageSize(2) = ImageSize(1); end
    if numel(ImageSize) < 3, ImageSize(3) = ImageSize(2); end
  else
    ImageSize = handles.ImageDim(1:3);
  end
end

handles.ImageSize = ImageSize;

if isempty(handles.projection)
  handles.projection = fix(handles.ImageDim/2);
else
  handles.projection(1) = min(handles.projection(1), handles.ImageDim(1));
  handles.projection(2) = min(handles.projection(2), handles.ImageDim(2));
  handles.projection(3) = min(handles.projection(3), handles.ImageDim(3));
  handles.projection(4) = min(handles.projection(4), handles.ImageDim(4));
end
handles.projection(handles.projection < 1) = 1;

% find image fourth dimentsion
Images = getappdata(handles.figure1, 'Images');
handles.RawImage = -1;
for ii=1:length(Images)
  if isequal(Images{ii}.Type, 'Raw'), handles.RawImage = ii; end
end
if handles.RawImage ~= -1
  LoadedImage.raw_info = safeget(LoadedImage, 'raw_info', []);
  LoadedImage.raw_info.data = safeget(LoadedImage.raw_info, 'data', []);
  Modality = safeget(LoadedImage.raw_info.data, 'Modality', '');
  switch Modality
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Pulse data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 'PULSEFBP'
      switch safeget(safeget(LoadedImage.raw_info, 'data', []), 'Sequence', '')
        case '3pT1'
          Images{handles.RawImage}.Dim4   = safeget(LoadedImage.raw_info, 'T1', [])*1E6;
          Images{handles.RawImage}.Dim4Unit = 'Stim. echo delay, us';
        case '2pECHO'
          Images{handles.RawImage}.Dim4   = safeget(LoadedImage.raw_info, 'tau2', [])*1E6;
          Images{handles.RawImage}.Dim4Unit = 'tau, us';
        case 'ESEInvRec'
          Images{handles.RawImage}.Dim4   = safeget(LoadedImage.raw_info, 'T1', [])*1E6;
          Images{handles.RawImage}.Dim4Unit = 'Inversion delay, us';
        case 'FIDInvRec'
          %             T1 = safeget(raw_info, 'T', []);
          %             axis_4D  = T1*1E6;
          %             axis_label = 'T1 [us]';
          %             if handles.plot4 ~=2
          %               fit_show = linspace(axis_4D(1), axis_4D(end), 50);
          %               [fit_amp, fit_t1, fit_inv, fit_err_mask, fit_error] = fit_recovery_3par(data(:)', axis_4D);
          %               fit_result = fit_amp - fit_inv*exp(-fit_show/fit_t1);
          %               rel_time = [fit_t1, fit_error(2)];
          %               rel_amp  = [fit_amp, fit_error(1)];
          %             end
        case 'FIDSRT'
          %             Trep = safeget(raw_info, 'Trep', []);
          %             axis_4D  = Trep;
          %             axis_label = 'Trep [us]';
        case 'Rabi'
          %             tp = safeget(raw_info, 'tp', []);
          %             axis_4D  = tp*1e9;
          %             axis_label = 'pulse [ns]';
        case '2pECHOSRT'
          %             Trep = safeget(raw_info, 'Trep', []);
          %             axis_4D  = [Trep; max(Trep)*4]*1E6;
          %             axis_label = 'Trep [us]';
        otherwise
          %             tau2 = safeget(raw_info, 'tau2', []);
          %             if ~isempty(tau2)
          %             axis_4D  = tau2 * 1E6;
          %             axis_label = 'tau*2 [us]';
          %             else
          %               axis_4D = 1:length(data);
          %               axis_label = 'N []';
          %             end
          %
          %             if handles.plot4 ~=2
          %               fit_show = linspace(axis_4D(1), axis_4D(end), 50);
          %               [fit_amp, fit_t2, fit_err_mask, fit_error] = fit_exp_no_offset([data(:)', 0], [tau2; max(tau2)*4]*1E6);
          %               fit_result = fit_amp * exp(-fit_show/fit_t2);
          %               rel_time = [fit_t2, fit_error(2)];
          %               rel_amp  = [fit_amp, fit_error(1)];
          %             end
          %         end
      end
    case 'RSFBP'
      deltaB = safeget(LoadedImage.rec_info.rec, 'deltaH', 1);
      Images{handles.RawImage}.Dim4   = linspace(-deltaB/2,deltaB/2, Images{handles.RawImage}.Dim(4));
      Images{handles.RawImage}.Dim4Unit = 'Field, G';
    case 'PULSESPI'
      switch safeget(safeget(LoadedImage.raw_info, 'data', []), 'Sequence', '')
          case 'FIDInvRec'
              Images{handles.RawImage}.Dim4   = safeget(LoadedImage.raw_info, 'T', [])*1E6;
              Images{handles.RawImage}.Dim4Unit = 'Inversion delay, us';
          case 'FID'
              Images{handles.RawImage}.Dim4   = safeget(LoadedImage.raw_info, 'T', [])*1E6;
              Images{handles.RawImage}.Dim4Unit = 'T2*, us';
      end
      otherwise
        if isfield(LoadedImage, 'Dim4')
          Images{handles.RawImage}.Dim4 = LoadedImage.Dim4;
        end
  end
else
  % this is timeline
  if isstruct(file_name) && isfield(file_name, 'StartTime') 
    Dim4 = file_name.StartTime' - min(file_name.StartTime);
    for ii=1:length(Images)
      Images{ii}.Dim4 = Dim4;
    end
  end
end
%       if ~isfield(handles.Image, 'raw_info'), handles.Image.raw_info.data.Modality = '?'; end
%       if length(data) == 1
%         plot(0, 0, 'Parent',hh); xlabel(hh,'');
%       elseif strcmp(Modality, 'PULSEFBP')
setappdata(handles.figure1, 'Images', Images);
setappdata(handles.figure1, 'ax4info', ax4info);

handles.options.size.Dim = handles.ImageDim(1:3);
handles.options.size.Size = handles.ImageSize(1:3);
handles.options.size.FOV  = handles.options.size.Size;
if isfield(LoadedImage, 'options')
  options = LoadedImage.options;
  if isfield(options, 'roi')
    handles.options.roi = options.roi;
  end
end

handles.NCxyz = [0,0,0];
handles.REFxyz = [0,0,0];
handles.A = GetA(handles);

guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function [ImageDim, options] = Data_Check(hGUI, DataMask, options)

% Find out image dimension
Images = getappdata(hGUI, 'Images');

ImageDim = [1,1,1];
for ii=1:length(Images)
  % Get Image size 
  sz = size(Images{ii}.Image);
  % Bring it to standard 4 element length
  if length(sz) == 2, sz(3) = 1; sz(4) = 1;
  elseif length(sz) == 3, sz(4) = 1;
  else
    sz = sz(1:4);
  end
  Images{ii}.Dim = sz;
  if ii==1 
    ImageDim = sz; 
  elseif ~isequal(sz(1:3), ImageDim(1:3))  
    error('Data_Check: dimensions of arrays do not match.');
  else
    sz(4) = ImageDim(4);
  end   
end

% Generate mask for the data 
Masks = getappdata(hGUI, 'Masks');
isNoMask = false;
if isempty(DataMask) || ~any(DataMask(:))
  DataMask = true(ImageDim(1:3));
  isNoMask = true;
end
Masks{1}.Mask = DataMask;
Masks{1}.Color = '-';
Masks{1}.Name = 'Data mask';
setappdata(hGUI, 'Masks', Masks);

osettings = safeget(options,'settings',[]);
for ii=1:length(Images)
  if Images{ii}.Dim(4) == 1
%        opt.minmax = double(epr_minmax(Images{ii}.Image));
      %commented to fix problem loading images with masks onboard
 opt.minmax = double(epr_minmax(Images{ii}.Image(DataMask(:,:,:,1))));
    opt.scale_minmax = opt.minmax;
    % try to remove outliers
    if ~isNoMask
%         opt.scale_minmax = double(epr_minmax(Images{ii}.Image, '0.05%'))
     opt.scale_minmax = double(epr_minmax(Images{ii}.Image(DataMask(:,:,:,1)), '0.05%'));
    end
    Images{ii}.Dim4 = [];
  else % RAW IMAGE
    opt.minmax = double(epr_minmax(Images{ii}.Image));
    opt.scale_minmax = opt.minmax;
    if ~isfield(Images{ii}, 'Dim4')
      Images{ii}.Dim4 = 1:Images{ii}.Dim(4);
    end
    Images{ii}.Dim4Unit = safeget(Images{ii}, 'Dim4Unit', []);
    Images{ii}.Dim4Image = safeget(Images{ii}, 'Dim4Image', []);
  end
  if isfield(osettings, Images{ii}.Type)
    settings = osettings.(Images{ii}.Type);
    Images{ii}.Unit = settings.field_unit;
    Images{ii}.Description = settings.field_dsc;
    Images{ii}.DefaultRange = settings.default_range;
  else
    Images{ii}.Unit = 'au';
    Images{ii}.Description = Images{ii}.Type;
    Images{ii}.DefaultRange = [0,1];
  end
  opt.isLowlim = 1;
  opt.isHighlim = 1;
  options.(Images{ii}.Type) = opt;
end

setappdata(hGUI, 'Images', Images);

% --------------------------------------------------------------------
function mFileLoad_Callback(hObject, eventdata, handles)

dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'SourcePath', 'C:/');

[FileName,PathName] = uigetfile({'2dseq;*.img;*.mat;*.tdms', 'All supported types (2dseq;*.img; *.mat)'; ...
  '*.mat', 'Matlab files (*.mat)'; '2dseq;*.img', 'Bruker files (2dseq;*.img)'; '*.tdms', 'OxyVue files (*.tdms)';'*.*', 'All files (*.*)'},'Load file', old_path);

if ~(isequal(FileName,0) || isequal(PathName,0))
  Data_Load(fullfile(PathName, FileName), handles, true, true);
  handles = guidata(handles.figure1);
  Mask_Check(handles.figure1, handles.ImageDim);
  UpdateMaskList(handles);
  AfterProjectionChanged(handles);
end

% --------------------------------------------------------------------
function mFileReLoad_Callback(hObject, eventdata, handles)
Data_Load(handles.options.FileName, handles, true, false);
handles = guidata(handles.figure1);
Mask_Check(handles.figure1, handles.ImageDim);
UpdateMaskList(handles);
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
function mFileLoadDB_Callback(hObject, eventdata, handles)
[filenames, isOk] = BuildSQLqueryMISDB;
if isOk
    Data_Load(filenames{1}, handles, true, true);
    handles = guidata(handles.figure1);
    Mask_Check(handles.figure1, handles.ImageDim);
    UpdateMaskList(handles);
    AfterProjectionChanged(handles);
end

% --------------------------------------------------------------------
function handles = UpdateProjection(handles)
image_number = get(handles.pmImageType, 'Value');
[image_type, image_dim] = Data_Stat(handles.figure1, image_number);

if ~isempty(handles.projection)
  edit_boxes = [handles.editDim1, handles.editDim2, handles.editDim3, handles.editDim4];
  for ii=1:4
    handles.projection(ii) = round(max(1, handles.projection(ii)));
    handles.projection(ii) = min(image_dim(ii), handles.projection(ii));
    handles.projection(handles.projection < 1) = 1;
    set(edit_boxes(ii), 'String', num2str(handles.projection(ii)));
  end
end
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function varargout = GetImageMask(handles, im_type)
Masks = getappdata(handles.figure1, 'Masks');

Mask = Masks{1}.Mask;

SelectedMask = Mask;
if ~isempty(SelectedMask)
  opt_mask = safeget(handles.options, 'mask', struct('erosion', 0));
  switch safeget(opt_mask, 'erosion', 0)
    case -1, SelectedMask = imerode(SelectedMask, epr_strel('sphere', 1));
    case -2, SelectedMask = imerode(SelectedMask, epr_strel('sphere', 2));
    case -3, SelectedMask = imerode(SelectedMask, epr_strel('sphere', 3));
    case -4, SelectedMask = imerode(SelectedMask, epr_strel('sphere', 4));
    case -5, SelectedMask = imerode(SelectedMask, epr_strel('sphere', 5));
  end
end

Images = getappdata(handles.figure1, 'Images');
for ii=1:length(Images)
  opt = handles.options.(Images{ii}.Type);
  stat_lim = safeget(opt, 'stat_lim', [0 1000]);
  if safeget(opt, 'isStatLowlim', 0)
    SelectedMask = SelectedMask & Images{ii}.Image >= stat_lim(1);
  end
  if safeget(opt, 'isStatHighlim', 0)
    SelectedMask = SelectedMask & Images{ii}.Image<= stat_lim(2);
  end
  
  show_lim = safeget(opt, 'show_lim', [0 1000]);
  if safeget(opt, 'isShowLowlim', 0)
    Mask = Mask & Images{ii}.Image >= show_lim(1);
  end
  if safeget(opt, 'isShowHighlim', 0)
    Mask = Mask & Images{ii}.Image <= show_lim(2);
  end
end

switch nargout
  case 1, varargout{1} = Mask;
  case 2, varargout{1} = Mask; varargout{2} = SelectedMask;
end

% --------------------------------------------------------------------
function [Min, Max] = GetScale(imtype, opt)

options = opt.(imtype);
if safeget(options, 'isLowlim', 0) || safeget(options, 'isHighlim', 0)
  scale_minmax = safeget(options, 'scale_minmax', options.minmax);
  Min = scale_minmax(1); Max = scale_minmax(2);
else
  Min  = options.minmax(1); Max  = options.minmax(2);
end

% --------------------------------------------------------------------
function [ax, i_dim] = ConvertAxis(a_start, a_end, A, n)
a_start = a_start * A; a_end = a_end * A;

[~, i_dim] = max(abs(a_start - a_end));
ax = linspace(a_start(i_dim), a_end(i_dim), n);

% --------------------------------------------------------------------
function Draw_Image(handles, planes, hh, ext_axis, options)
if ~exist('options','var'), options = handles.options; end

if exist('hh','var'), ext_figure = true; else, ext_figure = false; end
export_slice = safeget(handles, 'export_slice', false);

global last_hit_axis3D;

if handles.hit_ax < 4
  last_hit_axis3D = handles.hit_ax;
end

image_number = get(handles.pmImageType, 'Value');
[data, image_info] = Data_Get(handles.figure1, image_number);
if isempty(data)
  hh_ax = [handles.axes_1, handles.axes_2, handles.axes_3, handles.axesAdt];
  for ii=1:4
    cla(hh_ax(ii));
    text(0.5, 0.5, 'Data not loaded.', 'HorizontalAlignment', 'center', 'Parent', hh_ax(ii));
    axis(hh_ax(ii), [0 1 0 1]);
  end
  return;
end

isXRayView = safeget(handles.options,'isXRayView', false);
isMaskRaw = safeget(handles.options,'isMaskRaw', false);
isContoursRestrict = safeget(handles.options, 'isContoursRestrict', false);

yx_mask_add = {}; zx_mask_add = {}; yz_mask_add = {};

[Mask, SelMask] = GetImageMask(handles, image_info.image_type);
StatMask = epr_GetSphericMask(image_info.image_dim, handles.projection, handles.ROIrad);
if ~isMaskRaw && isequal(image_info.image_type, 'Raw')
  Mask = true(image_info.image_dim(1:3));
else
  if ndims(Mask) > 3
    Mask = Mask(:,:,:,min(end, planes(4)));
  end
end

% this speeds up a little for very large 3D matrices
if ndims(data) > 3
  data = data(:,:,:,min(end, planes(4)));
end

if isempty(SelMask), SelMask = true(sz(1:3)); end

% data(~Mask) = 0;

image_size_opt = safeget(handles.options, 'size', []);
image_mask_opt = safeget(handles.options, 'mask', struct('erosion', 0));

roi_struct = safeget(handles.options, 'roi', []);
FOV = iff(safeget(image_size_opt, 'isFOV', 0), safeget(image_size_opt, 'FOV', handles.ImageSize(1)), handles.ImageSize(1));
ROI = iff(safeget(roi_struct, 'isROI', 0), safeget(roi_struct, 'ROI', FOV), FOV);
[idx_roi] = epr_GetROIbins(FOV, ROI, size(Mask), planes);
planes_xyz = [planes(1:3), 1] / handles.A;

[x, xpos] = ConvertAxis([min(idx_roi.x), 0, 0, 1], [max(idx_roi.x), 0, 0, 1], inv(handles.A), length(idx_roi.x));
[y, ypos] = ConvertAxis([0, min(idx_roi.y), 0, 1], [0, max(idx_roi.y), 0, 1], inv(handles.A), length(idx_roi.y));
if ~isempty(idx_roi.z)
  [z, zpos] = ConvertAxis([0, 0, min(idx_roi.z), 1], [0, 0, max(idx_roi.z), 1], inv(handles.A), length(idx_roi.z));
else
  zpos = 0;
  z = 0;
end
handles.ax123info.idx = [xpos, ypos, zpos];
handles.ax123info.lim(1,:) = [x(1), x(end)];
handles.ax123info.lim(2,:) = [y(1), y(end)];
handles.ax123info.lim(3,:) = [z(1), z(end)];

handles.ax123info.max_lim = [];
handles.ax123info.max_lim{1} = ConvertAxis([1, 0, 0, 1], [image_info.image_dim(1), 0, 0, 1], inv(handles.A), image_info.image_dim(1));
handles.ax123info.max_lim{2} = ConvertAxis([0, 1, 0, 1], [0, image_info.image_dim(2), 0, 1], inv(handles.A), image_info.image_dim(2));
handles.ax123info.max_lim{3} = ConvertAxis([0, 0, 1, 1], [0, 0, image_info.image_dim(3), 1], inv(handles.A), image_info.image_dim(3));

[Min, Max] = GetScale(image_info.image_type, handles.options);
clim = [Min, Max];
ShowColorbarFIG(handles.axScale, 'fixed', clim, 'text', 'right')
set(handles.eShowMin, 'String', num2str(Min));
set(handles.eShowMax, 'String', num2str(Max));

[yx, zx, yz]=epr_getslice3D(data, planes([2,1,3]), idx_roi.x, idx_roi.y, idx_roi.z);
[yx_mask, zx_mask, yz_mask]=epr_getslice3D(Mask, planes([2,1,3]), idx_roi.x, idx_roi.y, idx_roi.z);

if handles.ROIrad ~= 0
  [yx_mask_add{end+1}.Mask, zx_mask_add{end+1}.Mask, yz_mask_add{end+1}.Mask]=...
    epr_getslice3D(StatMask, planes([2,1,3]), idx_roi.x, idx_roi.y, idx_roi.z);
  lw = 2; yx_mask_add{end}.LineWidth = lw;  zx_mask_add{end}.LineWidth = lw;  yz_mask_add{end}.LineWidth = lw;
  yx_mask_add{end}.Name = 'ROI cursor'; yx_mask_add{end}.Color = 'k';
  zx_mask_add{end}.Name = 'ROI cursor'; zx_mask_add{end}.Color = 'k';
  yz_mask_add{end}.Name = 'ROI cursor'; yz_mask_add{end}.Color = 'k';
end

if safeget(image_mask_opt, 'erosion', 0) ~= 0
  [yx_mask_add{end+1}.Mask, zx_mask_add{end+1}.Mask, yz_mask_add{end+1}.Mask]=...
    epr_getslice3D(SelMask, planes([2,1,3]), idx_roi.x, idx_roi.y, idx_roi.z);
  lw = 2; yx_mask_add{end}.LineWidth = lw;  zx_mask_add{end}.LineWidth = lw;  yz_mask_add{end}.LineWidth = lw;
end

if ~isempty(handles.TumorEllipsoid.Mask)
  [yx_mask_add{end+1}.Mask, zx_mask_add{end+1}.Mask, yz_mask_add{end+1}.Mask]=...
    epr_getslice3D(handles.TumorEllipsoid.Mask, planes([2,1,3]), idx_roi.x, idx_roi.y, idx_roi.z);
  lw = 2; yx_mask_add{end}.LineWidth = lw;  zx_mask_add{end}.LineWidth = lw;  yz_mask_add{end}.LineWidth = lw;
  cl = 'r'; yx_mask_add{end}.Color = cl;  zx_mask_add{end}.Color = cl;  yz_mask_add{end}.Color = cl;
end

if ~isempty(handles.TumorEllipsoid.Resonator)
  [yx_mask_add{end+1}.Mask, zx_mask_add{end+1}.Mask, yz_mask_add{end+1}.Mask]=...
    epr_getslice3D(handles.TumorEllipsoid.Resonator, planes([2,1,3]), idx_roi.x, idx_roi.y, idx_roi.z);
  lw = 2; yx_mask_add{end}.LineWidth = lw;  zx_mask_add{end}.LineWidth = lw;  yz_mask_add{end}.LineWidth = lw;
  cl = 'k'; yx_mask_add{end}.Color = cl;  zx_mask_add{end}.Color = cl;  yz_mask_add{end}.Color = cl;
end

% draw masks
Masks = getappdata(handles.figure1, 'Masks');
auto_colors = handles.auto_colors;
auto_colors = [auto_colors auto_colors auto_colors auto_colors auto_colors auto_colors auto_colors auto_colors auto_colors];
auto_c_idx = 1;
if ~isempty(Masks)
  for ii=1:length(Masks)
    if ~isempty(Masks{ii}.Mask) && ~isequal(Masks{ii}.Color, '-')
      MaskMask = Masks{ii}.Mask;
      if isContoursRestrict
        MaskMask = MaskMask & SelMask(:,:,:,1);
      end
      if isXRayView
        [yx_mask_add{end+1}.Mask, zx_mask_add{end+1}.Mask, yz_mask_add{end+1}.Mask]=...
          epr_getxray3D(MaskMask, planes([2,1,3]), idx_roi.x, idx_roi.y, idx_roi.z);
      else
        [yx_mask_add{end+1}.Mask, zx_mask_add{end+1}.Mask, yz_mask_add{end+1}.Mask]=...
          epr_getslice3D(MaskMask, planes([2,1,3]), idx_roi.x, idx_roi.y, idx_roi.z);
      end
      lw = safeget(Masks{ii}, 'LW', 2);
      if isequal(Masks{ii}.Color,'auto')
        cl = auto_colors{auto_c_idx};
      else
        cl = Masks{ii}.Color;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
      end
      yx_mask_add{end}.Name = Masks{ii}.Name; yx_mask_add{end}.Color = cl; yx_mask_add{end}.LineWidth = lw;
      zx_mask_add{end}.Name = Masks{ii}.Name; zx_mask_add{end}.Color = cl; zx_mask_add{end}.LineWidth = lw; 
      yz_mask_add{end}.Name = Masks{ii}.Name; yz_mask_add{end}.Color = cl; yz_mask_add{end}.LineWidth = lw;
    end
    auto_c_idx = auto_c_idx+1;
  end
end

draw4_planes = planes;
% Draw tracks
if handles.plot4 == 6
  SelMask = ImageInZAxis(SelMask, last_hit_axis3D);
  draw4_planes = ImageInZAxis(planes, last_hit_axis3D);
  
  r_track = 1;
  tracks = handles.tracks;
  tracks{end+1} = struct('planes', draw4_planes, 'hit_ax', last_hit_axis3D);
  for ii=1:length(tracks)
    track_mask = false(size(data));
    track_mask(tracks{ii}.planes(1)+r_track*(-1:1),tracks{ii}.planes(2)+r_track*(-1:1),:) =true;
    track_mask = ImageInNormalAxis(track_mask & SelMask, tracks{ii}.hit_ax);
    [yx_mask_add{end+1}.Mask, zx_mask_add{end+1}.Mask, yz_mask_add{end+1}.Mask]=...
      epr_getslice3D(track_mask, planes([2,1,3]), idx_roi.x, idx_roi.y, idx_roi.z);
    lw = 2; yx_mask_add{end}.LineWidth = lw;  zx_mask_add{end}.LineWidth = lw;  yz_mask_add{end}.LineWidth = lw;
    cl = 'm'; yx_mask_add{end}.Color = cl;  zx_mask_add{end}.Color = cl;  yz_mask_add{end}.Color = cl;
  end
end

options.clim = clim;
if image_info.image_type == 4, options.ShowGuide = 'g:'; else options.ShowGuide = 'k:'; end

options.ShowGuide = safeget(handles.options, 'guides', 'k:');
options.FontColor = safeget(handles.options, 'font_color', 'k');

plot_idx = Plot_Idx(options.plots_idx, handles.ax123info.idx);

% opt.colormap = [];
options.axis = iff(length(x)==length(y), 'square', 'equal');
% opt.box = 'off';
epr_DrawAxis(handles.axes_1, y, x, yx, yx_mask, yx_mask_add, planes_xyz(plot_idx{1}), options.plots_label{1}, options);
if ~isempty(idx_roi.z)
  epr_DrawAxis(handles.axes_2, z, x, zx, zx_mask, zx_mask_add, planes_xyz(plot_idx{2}), options.plots_label{2}, options);
  epr_DrawAxis(handles.axes_3, y, z, yz, yz_mask, yz_mask_add, planes_xyz(plot_idx{3}), options.plots_label{3}, options);
else
    cla(handles.axes_2);
    cla(handles.axes_3);
end

if export_slice
  handles.export.x = x; handles.export.y = y; handles.export.z = z;
  handles.export.yx = yx; handles.export.zx = zx; handles.export.yz = yz;
  handles.export.yx_mask = yx_mask; handles.export.zx_mask = zx_mask; handles.export.yz_mask = yz_mask;
end

[handles.ax4info, handles.export] = Draw_Panel4(handles, draw4_planes, SelMask, clim, handles.axesAdt, options);

if ext_figure
  switch ext_axis
    case 1, epr_DrawAxis(gca(hh), y, x, yx, yx_mask, yx_mask_add, planes_xyz(plot_idx{1}), options.plots_label{1}, options);
    case 2, epr_DrawAxis(gca(hh), z, x, zx, zx_mask, zx_mask_add, planes_xyz(plot_idx{2}), options.plots_label{2}, options);
    case 3, epr_DrawAxis(gca(hh), y, z, yz, yz_mask, yz_mask_add, planes_xyz(plot_idx{3}), options.plots_label{3}, options);
    case 4, Draw_Panel4(handles, draw4_planes, SelMask, clim, gca(hh), options);
  end
end

if isfield(options, 'background')
  set(hh,'color',options.background);
end

guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function [ax4info, export] = Draw_Panel4(handles, planes, data_mask, min_max, hh, options)
global last_hit_axis3D;
cMask = classMaskStorage(handles.figure1, 'Masks', options);

if ~exist('options','var'), options = handles.options; end

export_slice = safeget(handles, 'export_slice', false);
export = safeget(handles, 'export', []);

cla(hh); hold(hh, 'on');

Images  = getappdata(handles.figure1, 'Images');
ax4info = getappdata(handles.figure1, 'ax4info');
image_number = get(handles.pmImageType, 'Value');
if (handles.plot4 == 2 || handles.plot4 == 3) && handles.RawImage ~= -1
  image_number = handles.RawImage;
end
[data, image_info] = Data_Get(handles.figure1, image_number);

try
switch handles.plot4
  case 1 % 3D image profiles
    data = data(:,:,:,min(end, planes(4)));
%     mask = cMask.SafeGet(1, image_info.image_dim);
%     data(~mask) = 0;
    image_size_opt = safeget(handles.options, 'size', 64);
    FOV = iff(safeget(image_size_opt, 'isFOV', 0), safeget(image_size_opt, 'FOV', handles.ImageSize(1)), handles.ImageSize(1));
    ROI = iff(safeget(image_size_opt, 'isROI', 0), safeget(image_size_opt, 'ROI', FOV), FOV);
    [idx_roi, axis_roi] = epr_GetROIbins(FOV, ROI, image_info.image_dim, planes);
    
    dm_m = min_max(2) - min_max(1);
    zprj = squeeze(data(planes(1),planes(2),idx_roi.z));
    yprj = squeeze(data(planes(1),idx_roi.y,planes(3)));
    xprj = squeeze(data(idx_roi.x,planes(2),planes(3)));
    plot(axis_roi.z,zprj-dm_m*2*1.05, 'Parent', hh);
    plot(axis_roi.y,yprj-dm_m*1.05, 'Parent', hh);
    plot(axis_roi.x,xprj, 'Parent', hh)
    plot(axis_roi.z(planes(3)),zprj(planes(3))-dm_m*2*1.05, 'k.', 'Parent', hh);
    plot(axis_roi.y(planes(2)),yprj(planes(2))-dm_m*1.05, 'k.', 'Parent', hh);
    plot(axis_roi.x(planes(1)),xprj(planes(1)), 'k.', 'Parent', hh)
    %       set(handles.axesAdt, 'YTickLabel', {})
    text(-ROI*.45, min_max(1)+.2*dm_m, 'X', 'Parent', hh);
    text(-ROI*.45, min_max(1)+.2*dm_m-dm_m*1.05, 'Y', 'Parent', hh);
    text(-ROI*.45, min_max(1)+.2*dm_m-dm_m*2*1.05, 'Z', 'Parent', hh);
    %       if isfield(args, 'Title'), title(epr_ShortFileName(args.Title, 40), 'Interpreter', 'none'); end
    axis(hh, [-ROI/2, +ROI/2, min(zprj-dm_m*2*1.05)-dm_m*.1, min_max(2)+dm_m*.1]); box on
    xlabel(hh, ['Size [',Images{image_number}.Dim123Unit,']']);
%--------------------------------------------------------------------------     
  case 2 % 4th dimension of image
    if image_info.image_dim(4) > 1
      datay = squeeze(data(planes(1), planes(2), planes(3), :));
      if length(Images{image_number}.Dim4) == length(datay)
        plot(Images{image_number}.Dim4, datay, 'bo', 'Parent', hh); hold(hh, 'on');
        plot(Images{image_number}.Dim4(planes(4)), datay(planes(4)), 'm+', 'Parent', hh, 'MarkerSize', 10);
        plot(Images{image_number}.Dim4, zeros(size(datay)), ':', 'Parent', hh); hold(hh, 'on');
      else
        plot(datay, 'bo', 'Parent', hh); hold(hh, 'on');
      end
      min_max(1) = min(data(:));
      min_max(2) = max(data(:));
      axis(hh, [-Inf, +Inf, min_max(1), min_max(2)]); box on
      xlabel(hh, Images{image_number}.Dim4Unit);
    end
%--------------------------------------------------------------------------     
  case 3 % 4th dimension of image and fit
    RawImage = safeget(ax4info, 'RawImage', -1);
    if RawImage > 0
      data = squeeze(data(planes(1), planes(2), planes(3), :));
      xx = Images{RawImage}.Dim4;
      
      if length(data)<20
        plot(xx, data, 'bo', 'Parent', hh); hold(hh, 'on');
        plot(xx(planes(4)), data(planes(4)), 'm+', 'Parent', hh, 'MarkerSize', 10);
        plot(xx, zeros(size(data)), ':', 'Parent', hh); hold(hh, 'on');
      else
        plot(xx, data, 'b-', 'Parent', hh); hold(hh, 'on');
        plot(xx(planes(4)), data(planes(4)), 'm+', 'Parent', hh, 'MarkerSize', 10);
        plot(xx, zeros(size(data)), ':', 'Parent', hh); hold(hh, 'on');
      end
      min_max(1) = min(data(:));
      min_max(2) = max(data(:));
      axis(hh, [-Inf, +Inf, min_max(1), min_max(2)]); box on
      axis(hh, 'tight'); box on
      xlabel(hh, Images{image_number}.Dim4Unit);
      
      Modality=safeget(ax4info, 'Dim4Type', '');
      if length(data) == 1
        plot(0, 0, 'Parent',hh); xlabel(hh,'');
      elseif contains(Modality, 'Inversion')
        fit_array = data(:)'; fit_array(end+1) = fit_array(end);
        fit_xx = xx(:); fit_xx(end+1) = fit_xx(end)*2;
        [fit_amp, fit_t1, fit_inv, fit_err_mask, fit_error, ff] = fit_recovery_3par(fit_array, fit_xx);
        fit_show = linspace(xx(1), xx(end), 50);
        fit_result = ff([fit_amp, 1./fit_t1, fit_inv],fit_show);
        plot(fit_show, fit_result, '-', 'Parent', hh); hold(hh, 'on');
        text(.75, .22, sprintf('A=%5.2g', fit_amp), ...
          'Units', 'normalized', 'Parent', hh)
        text(.75, .12, sprintf('T=%4.1fus', fit_t1), ...
          'Units', 'normalized', 'Parent', hh)
        delta_po2 = epr_T2_PO2(1/(1/(fit_t1 - fit_error(2)))) - epr_T2_PO2(1/(1/(fit_t1)))
      elseif contains(Modality, 'T2Decay')
        [fit_amp, fit_t1, fit_err_mask, fit_error, ff] = fit_exp_no_offset(data(:)', xx(:));
        fit_show = linspace(xx(1), xx(end), 50);
        fit_result = ff([fit_amp, 1./fit_t1],fit_show);
        plot(fit_show, fit_result, '-', 'Parent', hh); hold(hh, 'on');
        text(.75, .22, sprintf('A=%5.2g', fit_amp), ...
          'Units', 'normalized', 'Parent', hh)
        text(.75, .12, sprintf('T=%4.1fus', fit_t1), ...
          'Units', 'normalized', 'Parent', hh)
      elseif contains(Modality, 'Stim')
        [fit_amp, fit_t1, fit_err_mask, fit_error, ff] = fit_exp_no_offset(data(:)', xx(:));
        fit_show = linspace(xx(1), xx(end), 50);
        fit_result = ff([fit_amp, 1./fit_t1],fit_show);
        plot(fit_show, fit_result, '-', 'Parent', hh); hold(hh, 'on');
        text(.75, .22, sprintf('A=%5.2g', fit_amp), ...
          'Units', 'normalized', 'Parent', hh)
        text(.75, .12, sprintf('T=%4.1fus', fit_t1), ...
            'Units', 'normalized', 'Parent', hh)
      elseif contains(Modality, 'T2*')
        [fit_amp, fit_t1, fit_err_mask, fit_error, ff] = fit_exp_no_offset(data(:)', xx(:));
        fit_show = linspace(xx(1), xx(end), 50);
        fit_result = ff([fit_amp, 1./fit_t1],fit_show);
        plot(fit_show, fit_result, '-', 'Parent', hh); hold(hh, 'on');
        text(.75, .22, sprintf('A=%5.2g', fit_amp), ...
          'Units', 'normalized', 'Parent', hh)
        text(.75, .12, sprintf('T=%4.2fus', fit_t1), ...
          'Units', 'normalized', 'Parent', hh)
      elseif contains(Modality, 'LORENTZIAN','IgnoreCase',true)
        mat_fit_info.spin_probe = 'Lorentzian';
        mat_fit_info.fit_method = 'lookup_general';
        mat_fit_info.fit_x0_min_max = linspace(-0.35,0.35, 27); % line offset
        mat_fit_info.fit_LW_min_max = linspace(0.05,0.65, 500);
        mat_fit_info.fit_ph_min_max = linspace(-10,10, 5)*pi/180; % phase
        
        [P, Perr] = fit_rs_amp_R2_phase_xover(xx(:), data(:), mat_fit_info);

        [fit_amp, fit_t1, fit_err_mask, fit_error, ff] = rs_spectral_fit(data(:)', xx(:));
        fit_show = linspace(xx(1), xx(end), 50);
        fit_result = ff([fit_amp, 1./fit_t1],fit_show);
        plot(fit_show, fit_result, '-', 'Parent', hh); hold(hh, 'on');
        text(.75, .22, sprintf('A=%5.2g', fit_amp), ...
          'Units', 'normalized', 'Parent', hh)
        text(.75, .12, sprintf('T=%4.2fus', fit_t1), ...
          'Units', 'normalized', 'Parent', hh)
        
      elseif 1==2
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Pulse data
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        switch safeget(safeget(raw_info, 'data', []), 'Sequence', '')
          case '3pT1'
            T1 = safeget(raw_info, 'T1', []);
            axis_4D  = T1*1E6;
            axis_label = 'T1 [us]';
            if handles.plot4 ~=2
              fit_show = linspace(axis_4D(1), axis_4D(end), 50);
              [fit_amp, fit_t1, fit_err_mask, fit_error] = fit_exp_no_offset(data(:)', axis_4D);
              fit_result = fit_amp * exp(-fit_show/fit_t1);
              rel_time = [fit_t1, fit_error(2)];
              rel_amp  = [fit_amp, fit_error(1)];
            end
          case 'ESEInvRec'
            TT   = safeget(raw_info, 'T1', [])*1E6;
            tau2 = 2*safeget(raw_info, 'tau', [])*1E6;
            Trep = safeget(raw_info, 'Trep', [])*1E6;
            tauT1 = tau2(1);
            idxT1 = tau2 == tauT1;
            idxT2 = ~idxT1;
            if length(idxT2) == 1
              axis_4D  = TT;
              axis_label = 'T [us]';
              if handles.plot4 ~=2
                fit_show = linspace(axis_4D(1), axis_4D(end), 50);
                [fit_amp, fit_t1, fit_inv, fit_err_mask, fit_error,ff] = fit_recovery_3par(data(:)', axis_4D(:));
                fit_result = ff([fit_amp, fit_t1, fit_inv],fit_show);
                rel_time = [fit_t1, fit_error(2)];
                rel_amp  = [fit_amp, fit_error(1)];
              end
            else
              axis_label = 'T, then T+tau*2 [us]';
              axis_4D = TT + tau2 - tau2(1);
              if handles.plot4 ~=2
                fit_showT1 = [linspace(min(TT), max(TT), 50), max(TT)*ones(1,20)];
                fit_showT2 = [min(tau2)*ones(1,50), linspace(min(tau2), 4*max(tau2), 20)];
                [tt,TR_for_show] = epr_DelaySpace(min(tau2), 4*max(tau2), 20, 'linear', min(Trep(idxT2)),6);
                fit_showTR = [linspace(min(Trep(idxT1)), max(Trep(idxT1)), 50),TR_for_show];
                fit_show   = [linspace(min(TT), max(TT), 50), max(TT)-min(tau2)+linspace(min(tau2), 4*max(tau2), 20)];
                [fit_amp, fit_t1, fit_inv, fit_t2, fit_err_mask, fit_error, ff] = fit_recovery_saturated(data(:)', tau2, TT, Trep);
                fit_result = ff([fit_amp, fit_t1, fit_inv, fit_t2],fit_showT2,fit_showT1,fit_showTR);
                rel_time = [fit_t1, fit_error(2)];
                rel_amp  = [fit_amp, fit_error(1)];
              end
            end
          case 'FIDInvRec'
            T1 = safeget(raw_info, 'T', []);
            axis_4D  = T1*1E6;
            axis_label = 'T1 [us]';
            if handles.plot4 ~=2
              fit_show = linspace(axis_4D(1), axis_4D(end), 50);
              [fit_amp, fit_t1, fit_inv, fit_err_mask, fit_error] = fit_recovery_3par(data(:)', axis_4D);
              fit_result = fit_amp - fit_inv*exp(-fit_show/fit_t1);
              rel_time = [fit_t1, fit_error(2)];
              rel_amp  = [fit_amp, fit_error(1)];
            end
          case 'FIDSRT'
            Trep = safeget(raw_info, 'Trep', []);
            axis_4D  = Trep;
            axis_label = 'Trep [us]';
          case 'Rabi'
            tp = safeget(raw_info, 'tp', []);
            axis_4D  = tp*1e9;
            axis_label = 'pulse [ns]';
          case '2pECHOSRT'
            Trep = safeget(raw_info, 'Trep', []);
            axis_4D  = [Trep; max(Trep)*4]*1E6;
            axis_label = 'Trep [us]';
          otherwise
            tau2 = safeget(raw_info, 'tau2', []);
            if ~isempty(tau2)
            axis_4D  = tau2 * 1E6;
            axis_label = 'tau*2 [us]';
            else
              axis_4D = 1:length(data);
              axis_label = 'N []';
            end
            
            if handles.plot4 ~=2
              fit_show = linspace(axis_4D(1), axis_4D(end), 50);
              [fit_amp, fit_t2, fit_err_mask, fit_error] = fit_exp_no_offset([data(:)', 0], [tau2; max(tau2)*4]*1E6);
              fit_result = fit_amp * exp(-fit_show/fit_t2);
              rel_time = [fit_t2, fit_error(2)];
              rel_amp  = [fit_amp, fit_error(1)];
            end
        end
        plot(axis_4D, data, 'bo', 'Parent', hh); hold(hh, 'on');
        
        if handles.plot4 ~=2
          plot(fit_show, fit_result, 'g-', 'Parent', hh)
          text(.95, .9, sprintf('T_2: %4.2f(+-%4.2f) us', rel_time(1), rel_time(2)), 'Units', 'normalized', ...
            'HorizontalAlignment', 'right', 'Parent', hh)
          text(.95, .7, sprintf('amp error %3.1f%%', rel_amp(2)*100/rel_amp(1)), 'Units', 'normalized', ...
            'HorizontalAlignment', 'right', 'Parent', hh)
          if isfield(handles.Image, 'pO2_info')
            text(.95, .8, sprintf('-> %4.1f(+-%4.1f) torr', epr_T2_PO2(rel_time(1), 0, 1, handles.Image.pO2_info),...
              rel_time(2)/rel_time(1)^2/pi/2/2.8*1000*handles.Image.pO2_info.Torr_per_mGauss), ...
              'Units', 'normalized', 'HorizontalAlignment', 'right', 'Parent', hh)
          end
        end
        axis(hh, 'tight'); xlabel(hh, axis_label);
        axx = axis(hh); axx(3) = min(axx(3), 0); axis(hh,axx);
      elseif strcmp(Modality, 'PULSEFBP')
        plot(data, 'bo', 'Parent', hh); hold(hh, 'on');
        xlabel(hh, safeget(handles.Image, 'Unit4Dim', ''));  axis(hh, 'tight');
        grid(hh, 'on');
      elseif strcmp(Modality, '?')
        plot(data, 'bo', 'Parent', hh); hold(hh, 'on');
        xlabel(hh, safeget(handles.Image, 'Unit4Dim', ''));  axis(hh, 'tight');
        grid(hh, 'on');
      else
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CW data and Rapid Scan
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fitdata = diff(data); fitdata = [fitdata(1); fitdata];
        if ~strcmp(Modality, 'RSFBP'), data = fitdata; end
        scan_width=safeget(handles.Image.rec_info.rec, 'deltaH', 1.024*sqrt(2));
        B = (1:length(data))' * scan_width / (length(data)- 0); B = B - mean(B);
        
        plot(B, data, 'b', 'Parent', hh); hold(hh, 'on');
        if handles.plot4 ~=2
          mat_fit_info = safeget(handles.Image, 'mat_fit_info', struct('spin_probe','OX063D24'));
          mat_fit_info.ModFrequency = safeget(raw_info, 'ModFrequency', 0.0001);
          mat_fit_info.ModAmplitude = safeget(raw_info, 'ModAmplitude', 0.0001);
          [pars, xx_err, out_fit] = fit_cw_amp_R2_phase_xover(B, fitdata, mat_fit_info);
          if strcmp(Modality, 'RSFBP'), out_fit = cumsum(out_fit); end
          
          plot(B, out_fit*pars(1), 'r', 'Parent', hh);
          plot(B, data-out_fit*pars(1), 'g', 'Parent', hh); hold(hh, 'off');
          text(.05, .2, sprintf('%4.1f(+-%4.1f) mG', pars(3)*1000, xx_err(3)*1000), 'Units', 'normalized', 'Parent', hh)
          if isfield(handles.Image, 'pO2_info')
            text(.05, .12, sprintf('%4.1f(+-%4.1f) torr', epr_LLW_PO2(pars(3)*1000, 0, 1, handles.Image.pO2_info),...
              xx_err(2)*1000*handles.Image.pO2_info.Torr_per_mGauss), ...
              'Units', 'normalized', 'Parent', hh)
          end
        end
        xlabel(hh, safeget(handles.Image, 'Unit4Dim', ''));  axis(hh, 'tight');
        grid(hh, 'on');
      end
    end
%--------------------------------------------------------------------------     
  case {4, 5} % statistics for all masks
    Masks = getappdata(handles.figure1, 'Masks');
    dset = cell(length(Masks));
    auto_colors = handles.auto_colors; auto_c_idx = 1;
    if image_info.image_dim(4) > 1, data = data(:,:,:,planes(4)); end
    for ii=1:length(Masks)
      if ii==1 || ~isequal(Masks{ii}.Color, '-')
        if ii==1
          [~, MaskMask] = GetImageMask(handles);
        else
          MaskMask = Masks{ii}.Mask;
        end
        if isempty(MaskMask), MaskMask = false(image_dim(1:3)); end
        if safeget(handles.options, 'isContoursRestrict', false)
          MaskMask = MaskMask & data_mask(:,:,:,1);
        end
        if handles.plot4 == 4
          stat_data = data(MaskMask==1);
          dset{ii}.show_mask = safeget(Masks{ii}, 'MakeStat', 1);
        else
          StatMask = epr_GetSphericMask(image_info.image_dim(1:3), handles.projection, handles.ROIrad);
          stat_data = data(MaskMask&StatMask);
          dset{ii}.show_mask = 1;
        end
        dset{ii}.stat_data = stat_data(stat_data >= min_max(1) & stat_data <= min_max(2));
        dset{ii}.mean = mean(dset{ii}.stat_data);
        dset{ii}.median = median(dset{ii}.stat_data);
        dset{ii}.std = std(dset{ii}.stat_data);
        if isequal(Masks{ii}.Color, 'auto') || isequal(Masks{ii}.Color, '-')
          dset{ii}.color = auto_colors{auto_c_idx}; 
          auto_c_idx = auto_c_idx + 1;
        else
           dset{ii}.color = Masks{ii}.Color;
        end
      end
    end
    
    if min_max(2) > 1, data_format = '%4.2f';
    elseif min_max(2) > 0.1, data_format = '%5.4f';
    elseif min_max(2) > 0.01, data_format = '%6.5f';
    else data_format = '%7.6g';
    end
    edges = linspace(min_max(1), min_max(2), 80);
    dx = mean(diff(edges));
    edges_plus = [min_max(1)-dx,edges,min_max(2)+dx];

    setpos = [.97, .77, .57, .37,  .17];
    hy = zeros(length(edges), length(dset));
    pp = (0:100) * 0.045;
    for jj=1:length(dset)
      if ~dset{jj}.show_mask, continue; end
      text(.97, setpos(jj)-pp(1), sprintf(['mn:', data_format], dset{jj}.mean), 'Units', 'normalized', 'HorizontalAlignment', 'right', 'Parent', hh, 'Color', dset{jj}.color)
      text(.97, setpos(jj)-pp(2), sprintf(['mdn:', data_format], dset{jj}.median), 'Units', 'normalized', 'HorizontalAlignment', 'right', 'Parent', hh, 'Color', dset{jj}.color)
      text(.97, setpos(jj)-pp(3), sprintf(['std:', data_format], dset{jj}.std), 'Units', 'normalized', 'HorizontalAlignment', 'right', 'Parent', hh, 'Color', dset{jj}.color)
      text(.97, setpos(jj)-pp(4), sprintf('n:%i', length(dset{jj}.stat_data)), 'Units', 'normalized', 'HorizontalAlignment', 'right', 'Parent', hh, 'Color', dset{jj}.color)
      hy(:, jj) = histc(dset{jj}.stat_data, edges);
      %       hhh = bar(edges,hy(:, jj), 'stacked','BarWidth',1, 'ShowBaseLine', 'off', 'Parent', hh);
      %       set(hhh, 'FaceColor', dset{jj}.color);
      %       bar(edges, hy(:, jj), dset{jj}.color, 'BarWidth', 1, 'ShowBaseLine', 'off', 'Parent', hh);
      fill(edges_plus, [0;hy(:, jj);0], dset{jj}.color, 'Parent', hh);
    end
    %     hy = flipdim(hy, 2);
    %     hhh=bar(edges,hy, 'stacked','BarWidth',1, 'ShowBaseLine', 'off', 'Parent', hh);
    %     set(hhh(1), 'FaceColor', 'r')
    %     set(hhh(2), 'FaceColor', 'b')
    
    if export_slice
      export.stat.yy = hy;
      export.stat.edges = edges;
      export.stat.raw = dset{1}.stat_data;
    end
    
    axis(hh, 'tight'); box(hh, 'on');
    hh.FontSize = safeget(options, 'axis4FontSize', 14);
    
    xlabel(hh, [Images{image_number}.Description, ' [', Images{image_number}.Unit,']']);    
%--------------------------------------------------------------------------     
  case 6 % show track
    ax4info.hit_ax = last_hit_axis3D;
    
    cla(hh)
    ROIMask = epr_GetSphericMask(Images{image_number}.Dim(1:3), handles.projection, handles.ROIrad);
    ROIMask = ImageInZAxis(ROIMask, ax4info.hit_ax);
    data = ImageInZAxis(data, ax4info.hit_ax);
    
    data_mask2D = sum(data_mask, 3) > 4 & sum(ROIMask, 3);
    ndata_mask2D = numel(find(data_mask2D(:)));
    idxZ = squeeze(any(any(data_mask, 1), 2));
    nidxZ = numel(find(idxZ));
    [ax4info.idx1, ax4info.idx2] = ind2sub(size(data_mask2D), find(data_mask2D));
    ax4info.idx3 = find(idxZ);
    ZScale = handles.ax123info.max_lim{handles.options.plots_idx1(ax4info.hit_ax)}(idxZ);
    ax4info.min_max = [min(ZScale), max(ZScale)];

    if ndata_mask2D > 0 && nidxZ > 0
      n1 = find(idxZ, 1, 'first');
      position = zeros(size(data_mask2D));
      position(planes(1), planes(2)) = 1;
      cidx = find(position(data_mask2D(:)));
      sz = size(data);
      tracks = reshape(data(:, :, idxZ), [sz(1)*sz(2), nidxZ]);
      track_masks = reshape(data_mask(:, :, idxZ), [sz(1)*sz(2), nidxZ]);
      tracks = tracks(data_mask2D(:), :);
      track_masks = track_masks(data_mask2D(:), :);
      nrep = max(fix(ndata_mask2D/5 + 0.5),1);
      if isempty(cidx), cidx = 1; end
      tracks = tracks([cidx + zeros(1, nrep), 1:ndata_mask2D], :);
      track_masks = track_masks([cidx + zeros(1, nrep), 1:ndata_mask2D], :);
      
      opt.clim = min_max;
      
      epr_DrawAxis(hh, -nrep+1:ndata_mask2D, linspace(ax4info.min_max(1), ax4info.min_max(2), nidxZ), ...
        tracks', track_masks', [], [], '', opt)
      axis(hh, 'normal');
      
      lw = 1.5;
      plot(0*[1,1], ax4info.min_max, 'k', 'Parent', hh, 'LineWidth', lw)
      plot(cidx*[1,1], ax4info.min_max, 'm', 'Parent', hh, 'LineWidth', lw)
      plot([-nrep+1, ndata_mask2D], ax4info.min_max(1) + (ax4info.min_max(2) - ax4info.min_max(1))*(planes(3)-n1+0.6)/nidxZ+[0,0], 'm', 'Parent', hh, 'LineWidth', lw);
    end
%--------------------------------------------------------------------------     
  case 7 % Timeline
    [data, image_info] = Data_Get(handles.figure1, image_number);
    if ndims(data) >= 4
      %     mask = Mask_Get(handles.figure1, 1, image_info.image_dim);
      
      sz = size(data);
      StatMask = epr_GetSphericMask(sz(1:3), handles.projection, handles.ROIrad);
      data4D = zeros(1, sz(4)); data4Derr = data4D;
      for ii=1:sz(4)
        datax = data(:,:,:,ii); 
        clip_max = datax > -98;
        subset_data = datax(StatMask(:) & clip_max(:));
        data4D(ii) = mean(subset_data); 
        data4Derr(ii) = std(subset_data); 
      end
      TimeLine = safeget(image_info, 'StartTime', []);
      if ~isempty(TimeLine)
        TimeLine = (TimeLine - TimeLine(1)) * 24 * 60;
        xlabel(hh, 'Time [min]')
      else
        TimeLine = 0:length(data4D)-1;
        xlabel(hh, 'N image')
      end
      if handles.ROIrad > 0
        errorbar(TimeLine, data4D, data4Derr, data4Derr, 'bo', 'Parent', hh); hold(hh, 'on');
      else
        plot(TimeLine, data4D, 'bo', 'Parent', hh); hold(hh, 'on');
      end
      plot(TimeLine(handles.projection(4))*[1,1],min_max,'m:', 'Parent', hh);
      %     plot(TimeLine, epr_median_filter(data4D,7,5), 'm-', 'Parent', hh,'linewidth',1.5);
      %     plot(TimeLine, data4D, 'm-', 'Parent', hh);
      axis(hh, [TimeLine(1),TimeLine(end),min_max(1:2)]);
      Excitation = safeget(image_info, 'Excitation', []);
      if ~isempty(Excitation)
        max_timeline = max(Excitation);
        min_timeline = min(Excitation);
        d_timeline = (max_timeline - min_timeline) / diff(min_max) * 2;
        plot(TimeLine, (Excitation(1:length(TimeLine)) - min_timeline)/d_timeline + min_max(1) + diff(min_max)*0.25, 'r-', 'Parent', hh)
      end
    end
  case 8 % sum of 4th dimension
    if ndims(data) >= 4
      data = squeeze(max(max(max(data, [], 1), [], 2), [], 3));
      plot(data, 'b-', 'Parent', hh);
      axis(hh, 'tight'); box(hh, 'on');
    end
end

if isfield(options, 'background')
  set(hh,'color',options.background);
end

catch err
  error_catcher('Draw_Panel4', err)
end

% --------------------------------------------------------------------
function error_catcher(func_destination, err)
for ii=1:length(err.stack)
  if isequal(func_destination, err.stack(ii).name)
    disp(sprintf('Error %s %s line %i: %s', iff(ii==1, 'in','in subfunction of'), func_destination, err.stack(ii).line, err.message));
  break;
  end
end

% --------------------------------------------------------------------
function y = simple_fit(x, pars)

lw = cw_robinson(pars.Bshf, x(1), x(2), pars.mod_omega, pars.mod_amp, 1);
y = conv2(imag(lw(:,1)*exp(-1i*x(3))),pars.pattern,'same');
y = interp1(pars.Bshf, y, pars.B,'spline');

% --------------------------------------------------------------------
function Draw_ROI_Info(handles)

pl = handles.projection;

str{1} = safeget(handles.options, 'description', '');
Images = getappdata(handles.figure1, 'Images');
for ii=1:length(Images)
  if isfield(handles.options.settings, Images{ii}.Type)
    if length(Images{ii}.Description) > 6 
       description = Images{ii}.Description(1:6); 
    else, description = Images{ii}.Description;
    end
    str{end+1} = sprintf('%6s = %7.2f %s', description, ...
        Images{ii}.Image(pl(1), pl(2), pl(3), min(pl(4), end)), Images{ii}.Unit);
  end
end

% for ii=1:length(handles.options.field_def)
%   if isfield(handles.Image, handles.options.field_def{image_type}) && ...
%       ~isempty(handles.Image.(handles.options.field_def{image_type})) && image_type == ii
%     if handles.ROIrad == 0
%       str{end+1} = sprintf('%s = %4.3g', handles.options.field_def{image_type}, ...
%         handles.Image.(handles.options.field_def{image_type})(pl(1), pl(2), pl(3), 1));
%     else
%       try
%         StatMask = epr_GetSphericMask(size(handles.Image.Mask), handles.projection, handles.ROIrad);
%         data = handles.Image.(handles.options.field_def{image_type})(handles.Image.Mask&StatMask);
%         if any(data(:))
%           str{end+1} = sprintf('M(%i)=%4.3g %s', handles.ROIrad, mean(data(:)),handles.options.field_unit{image_type});
%         else
%           str{end+1} = sprintf('%s = %4.3g', handles.options.field_def{image_type}, ...
%             handles.Image.(handles.options.field_def{image_type})(pl(1), pl(2), pl(3), pl(4)));
%         end
%       catch err
%         str{end+1} = sprintf('%s = %4.3g', handles.options.field_def{image_type}, ...
%           handles.Image.(handles.options.field_def{image_type})(pl(1), pl(2), pl(3), 1));
%       end
%     end
%     break;
%   end
% end
% 
% if isfield(handles.Image, 'Amp') && ~isempty(handles.Image.Amp)
%   str{end+1} = sprintf('Amp = %4.2f mM', handles.Image.Amp(pl(1), pl(2), pl(3)));
% end
% 
% if isfield(handles.Image, 'pO2') && ~isempty(handles.Image.pO2)
%   str{end+1} = sprintf('pO2 = %4.1f torr', handles.Image.pO2(pl(1), pl(2), pl(3)));
% end

set(handles.eLogWindow, 'String', str)

% --------------------------------------------------------------------
function pm4thplot_Callback(hObject, eventdata, handles) 
handles.plot4 = get(hObject, 'Value');
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
function eShowMinMax_Callback(hObject, eventdata, handles) 
image_number = get(handles.pmImageType, 'Value');
image_type = Data_Stat(handles.figure1, image_number);
val_low = str2double(get(handles.eShowMin, 'String'));
val_high = str2double(get(handles.eShowMax, 'String'));

handles.options.(image_type).scale_minmax = [val_low, val_high];
handles.options.(image_type).lim = [val_low, val_high];
AfterProjectionChanged(handles);
UpdateMultipleGUIoptions(hObject, eventdata, handles);

% --------------------------------------------------------------------
function pbDefaultRange_Callback(hObject, eventdata, handles) 
image_number = get(handles.pmImageType, 'Value');
image_type = Data_Stat(handles.figure1, image_number);
if isfield(handles.options.settings, image_type)
  default_range = handles.options.settings.(image_type).default_range;
  if isempty(default_range), return; end
  
  handles.options.(image_type).scale_minmax = default_range;
  handles.options.(image_type).lim = default_range;
  AfterProjectionChanged(handles);
  UpdateMultipleGUIoptions(hObject, eventdata, handles);
else
  disp(['Image ',image_type,' is not the standart type']);
end

% --------------------------------------------------------------------
function pbResetRange_Callback(hObject, eventdata, handles) 
image_number = get(handles.pmImageType, 'Value');
image_type = Data_Stat(handles.figure1, image_number);
reset_range = handles.options.(image_type).minmax;
  
handles.options.(image_type).scale_minmax = reset_range;
handles.options.(image_type).lim = reset_range;
AfterProjectionChanged(handles);
UpdateMultipleGUIoptions(hObject, eventdata, handles);

% --------------------------------------------------------------------
function pb0_Callback(hObject, eventdata, handles) 
image_number = get(handles.pmImageType, 'Value');
image_type = Data_Stat(handles.figure1, image_number);
  
handles.options.(image_type).scale_minmax(1) = 0;
AfterProjectionChanged(handles);
UpdateMultipleGUIoptions(hObject, eventdata, handles);

% --------------------------------------------------------------------
function UpdateMultipleGUIoptions(hObject, ~, handles) 
if get(handles.cbRange12321, 'Value')
  image_number = get(handles.pmImageType, 'Value');
  [image_type] = Data_Stat(handles.figure1, image_number);
  h = GetOtherGUI(handles);
  
  for ii=1:length(h)
    hh = guidata(h(ii));
    hh.options.(image_type).scale_minmax = ...
      handles.options.(image_type).scale_minmax;
    hh.options.(image_type).lim = ...
      handles.options.(image_type).lim;
    AfterProjectionChanged(hh, 0);
  end
end

% --------------------------------------------------------------------
function mOptionsOpt_Callback(hObject, ~, handles) 
% prepare list of options to manage
Images = getappdata(handles.figure1, 'Images');
option_list = cell(length(Images), 1);
for ii=1:length(Images)
  option_list{ii} = Images{ii}.Type;
end
handles.options.ActiveFields = option_list;

[opt_opt, res] = ImageCompareOptionsDLG(handles.options);
if res == 0 || isempty(opt_opt), return; end

opt_opt.size = opt_opt.size{1};

if res
  handles.options = opt_opt;
  handles.TumorEllipsoid.A = TumorTouchTransformation(handles.TumorEllipsoid, ...
    handles.ImageDim(1:3), handles.options.size.FOV);
  handles.A = handles.TumorEllipsoid.A;
  guidata(handles.figure1, handles);
  AfterProjectionChanged(handles);
end

% --------------------------------------------------------------------
function mMeasureDistance_Callback(hObject, ~, handles) 
a = ginput(2);
set(handles.eLogWindow, 'String', sprintf('The distance is\n %4.2f (%4.2f, %4.2f)', norm(a(1,:)-a(2,:)),...
  abs(a(1,1)-a(2,1)), abs(a(1,2)-a(2,2))))

% --------------------------------------------------------------------
function mOptionsCW_Callback(hObject, ~, handles) 
[handles.options.CWidx,v] = listdlg('PromptString','Select a file:',...
  'SelectionMode','multiple',...
  'ListString',handles.options.CWload, 'InitialValue', handles.options.CWidx);
if v, guidata(handles.figure1, handles); end

% --------------------------------------------------------------------
function mOptionsESE_Callback(hObject, ~, handles) 
[handles.options.ESEidx,v] = listdlg('PromptString','Select a file:',...
  'SelectionMode','multiple',...
  'ListString',handles.options.ESEload, 'InitialValue', handles.options.ESEidx);
if v, guidata(handles.figure1, handles); end

% --------------------------------------------------------------------
function mWindowNew_Callback(hObject, ~, handles) 
ibGUI([], handles.options);

% --------------------------------------------------------------------
function mWindowCopy_Callback(hObject, ~, handles) 

flds = {'ini','output','options','plot4','ROIrad','hit_ax','ax123info','ax4info','TumorEllipsoid','REFxyz','NCxyz','tracks','A',...
    'auto_colors','MaskToolBoxPos','projection','Toolbars','op_mode','op_mode_busy','StartFunction','EndFunction','ImageDim','is2D','ImageSize',...
    'RawImage', 'export', 'Timeline'};

transfer.transfer = 1;
transfer.Images = getappdata(handles.figure1, 'Images');
transfer.Masks = getappdata(handles.figure1, 'Masks');
for ii=1:length(flds)
  if isfield(handles, flds{ii})
    transfer.(flds{ii}) = handles.(flds{ii});
  end
end

ibGUI(transfer);

% --------------------------------------------------------------------
function mMeasureSNR_Callback(hObject, ~, handles) 
Raw = safeget(handles.Image, 'Raw', []);
if isempty(Raw), return; end

max_raw = max(Raw(:));
Mask = Raw > 0.2*max_raw;
Mask = imdilate(Mask, epr_strel('sphere', 2));
MaskMax = mean(Raw(Raw > 0.9*max_raw));

set(handles.eLogWindow, 'String', sprintf('SNR %4.1f', MaskMax/std(Raw(~Mask(:)))))
% ibGUI(Mask)

% --------------------------------------------------------------------
function mFileSave_Callback(hObject, ~, handles) 

[FileName,PathName, fidx] = uiputfile({'*.dcm', 'DICOM file (*.dcm)'; '*.mat', 'pMAT (*.mat)'},'Save file', handles.options.PathName);

if ~(isequal(FileName,0) || isequal(PathName,0))
  FileName = fullfile(PathName, FileName);
  
  switch fidx
      case 1
        dicomwritevolume(FileName, handles.Image.Amp);
      case 2
        [~, fit_mask] = GetImageMask(handles, 1);
        amp_dat = handles.Image.Amp(fit_mask);
        pO2_dat = handles.Image.pO2(fit_mask);
        Torr_per_mGauss = safeget(handles.Image.pO2_info, 'Torr_per_mGauss', 1.84);
        LLW_zero_po2    = safeget(handles.Image.pO2_info, 'LLW_zero_po2', 10.2);
        fit_data.P = zeros(4, length(amp_dat));
        fit_data.Perr = zeros(5, length(amp_dat));
        fit_data.Algorithm = 'T2T1_InvRecovery_3Par';
        fit_data.Size = size(handles.Image.Amp);
        fit_data.Parameters = {'Amplitude';'T1';'T2';'Inversion'};
        fit_data.Idx = find(fit_mask)';
        fit_data.FitMask = true(size(fit_data.Idx));
        fit_data.Mask =  true(size(fit_data.Idx));
        fit_data.P(1,:) = amp_dat*handles.Image.pO2_info.amp1mM;
        R2 = (pO2_dat/Torr_per_mGauss + LLW_zero_po2)*(pi*2*2.8)/1000; % mG -> s-1

        fit_data.P(2,:) = 1./R2;
        s.file_type    = 'FitImage_v1.1';
        s.raw_image    = handles.Image.SourceFileName;
        s.source_image = handles.Image.SourceFileName;
        s.raw_info     = handles.Image.raw_info;
        s.fit_data      = fit_data;
        s.rec_info     = handles.Image.rec_info;
        s.pO2_info     = handles.Image.pO2_info;
        save(FileName,'-struct','s');
  end  
end

% --------------------------------------------------------------------
function mWinAxis1_Callback(hObject, ~, handles) 
%%function [ax4info, export] = Draw_Panel4(handles, image_type, planes, data, data_mask, min_max, hh)
hh = figure;

options_axis = handles.options;
options_axis.colorbar = true;
options_axis.colorbarFont = 12;
options_axis.background = 'w';
options_axis.colormap = 'jet';

options_stat = handles.options;
options_stat.axis4FontSize = 24;
options_stat.background = 'w';
options_stat.figure_size = [560, 420];

switch hObject
  case handles.mWinAxis1
    Draw_Image(handles, handles.projection, hh, 1, options_axis)
  case handles.mWinAxis2
    Draw_Image(handles, handles.projection, hh, 2, options_axis)
  case handles.mWinAxis3
    Draw_Image(handles, handles.projection, hh, 3, options_axis)
  case handles.mWinAxis4
    Draw_Image(handles, handles.projection, hh, 4, options_stat)
    pp = get(gcf, 'position'); pp(3:4)=options_stat.figure_size;
    set(gcf, 'position',pp)
end

% --------------------------------------------------------------------
function mWindowSliceOmatic_Callback(hObject, ~, handles) 
image_number = get(handles.pmImageType, 'Value');
data = Data_Get(handles.figure1, image_number);

sliceomatic(double(data(:,:,:,1)))

% --------------------------------------------------------------------
function mOptionsView_Callback(hObject, ~, handles) 
hh = [handles.mOptionsView0, handles.mOptionsView1, handles.mOptionsView2, handles.mOptionsView3, handles.mOptionsView4, ...
  handles.mOptionsView5, handles.mOptionsView6];
set(hh, 'Checked', 'off'); set(hObject, 'Checked', 'on');
handles.ROIrad = find(hh==hObject) - 1; if isempty(handles.ROIrad), handles.ROIrad = 0; end
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
function mFileExport_Callback(hObject, ~, handles) 
export.Masks = getappdata(handles.figure1, 'Masks');
export.Images = getappdata(handles.figure1, 'Images');

assignin('base', 'export', export);
disp('Variable ''export'' is created in the workspace.');
disp(export);

% --------------------------------------------------------------------
function mFilePrint_Callback(hObject, ~, handles) 
% pagesetupdlg(handles.figure1);
set(handles.figure1, 'PaperPositionMode', 'auto')
printpreview(handles.figure1)
% printdlg(handles.figure1)

% --------------------------------------------------------------------
function im = ImageInZAxis(im, ax)
switch ndims(im)
  case {1, 2}
    switch ax
      case 3, im = im([2,3,1]);
      case 2, im = im([3,1,2]);
      case 1  % do nothing
    end
  case {3, 4}
    switch ax
      case 3, im = permute(im, [2,3,1]);
      case 2, im = permute(im, [3,1,2]);
      case 1  % do nothing
    end
end

% --------------------------------------------------------------------
function im = ImageInNormalAxis(im, ax)
switch ndims(im)
  case {1, 2}
    switch ax
      case 3, im = im([3,1,2]);
      case 2, im = im([2,3,1]);
      case 1  % do nothing
    end
  case {3, 4}
    switch ax
      case 3, im = permute(im, [3,1,2]);
      case 2, im = permute(im, [2,3,1]);
      case 1  % do nothing
    end
end

% --------------------------------------------------------------------
function mAddTT_Callback(hObject, ~, handles) 
TumorEllipsoid = safeget(handles, 'TumorEllipsoid', []);

[TumorEllipsoid, isOk] = TumorTouchEditDLG(TumorEllipsoid);

if isOk
  image_number = get(handles.pmImageType, 'Value');
  image_type = handles.Image_Idx(image_number);
  data = Data_Get(handles, image_type);
  image_size_opt = safeget(handles.options, 'size');
  FOV = iff(safeget(image_size_opt, 'isFOV', 0), safeget(image_size_opt, 'FOV', handles.Image.Size(1)), handles.Image.Size(1));
  sz = size(data);
  
  handles.TumorEllipsoid = TumorEllipsoid;
  handles.TumorEllipsoid.A = TumorTouchTransformation(TumorEllipsoid, sz(1:3), FOV*[1,1,1]);
  handles.A = handles.TumorEllipsoid.A;
  if TumorEllipsoid.expInfo.magnetType ~= 0
    if ~isempty(TumorEllipsoid.surfPts)
      surfPts_vox = htransform_vectors(handles.TumorEllipsoid.A, TumorEllipsoid.surfPts(:,1:3));
      penalty = TumorEllipsoid.surfPts(:,4);
      
      %     handles.TumorEllipsoid.Mask = false(sz(1:3));
      %     surfPts_vox = fix(surfPts_vox);
      %     for ii=1:size(TumorEllipsoid.surfPts, 1)
      %       handles.TumorEllipsoid.Mask(surfPts_vox(ii,1), surfPts_vox(ii,2), :) = true;
      %     end
      
      sizeTumor = safeget(TumorEllipsoid.expInfo, 'sizeTumor', [1,1,1]);
      n=sz(1); d=FOV/(n-1);
      R = 0.5*sizeTumor([1,2,3])/d;   % sizeTumor => radii of ellipsoid
      
      x = TumorTouchFitEllipsoid(sz(1:3), R, surfPts_vox, penalty);
      handles.TumorEllipsoid.Mask = TumorTouchEllipsoidMask(sz(1:3), x, R);
    else
      handles.TumorEllipsoid.Mask = [];
    end
    resonType = safeget(TumorEllipsoid.expInfo, 'resonType', '');
    sizeReson = epr_GetResonatorSize(resonType);
    
    gap_pix = [handles.TumorEllipsoid.expInfo.rGap, 1]*handles.TumorEllipsoid.A;
    
    handles.TumorEllipsoid.Resonator = MakeLGRMask(sizeReson, TumorEllipsoid.expInfo.magnetType, gap_pix, sz(1:3), FOV*[1,1,1]);
    
  else
    handles.TumorEllipsoid.Mask = [];
    handles.TumorEllipsoid.Resonator = [];
  end
  AfterProjectionChanged(handles);
end

% --------------------------------------------------------------------
function mAddOnBiopsyAdd_Callback(hObject, eventdata, handles) 
global last_hit_axis3D;
handles.tracks{end+1} = struct('planes', ImageInZAxis(handles.projection, handles.hit_ax), 'hit_ax', last_hit_axis3D);
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
function mAddOnBiopdyClearAll_Callback(hObject, eventdata, handles) 
handles.tracks = {};
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
function mAddOnBiopsyStat_Callback(hObject, eventdata, handles) 
global last_hit_axis3D;

tracks = handles.tracks;
tracks{end+1} = struct('planes', handles.projection(1:3), 'hit_ax', last_hit_axis3D);

image_number = get(handles.pmImageType, 'Value');
[data, data_info] = Data_Get(handles.figure1, image_number);

if isempty(data), return; end

[Mask, SelMask] = GetImageMask(handles, data_info.image_type);
data = ImageInZAxis(data, last_hit_axis3D);
Mask = ImageInZAxis(Mask, last_hit_axis3D);
SelMask = ImageInZAxis(SelMask, last_hit_axis3D);
r_track = 1;

for ii=1:length(tracks)
  if ii == length(tracks), disp('Current track'); else disp(sprintf('Track %i', ii)); end
  track  = tracks{ii};
  track.planesXYZ = htransform_vectors(inv(handles.A), track.planes);
  if ~isempty(data)
    track_mask = false(size(data));
    planesZ = ImageInZAxis(track.planes, last_hit_axis3D);
    track_mask(planesZ(1)+r_track*(-1:1),planesZ(2)+r_track*(-1:1),:) =true;
    track_mask = track_mask & Mask;
    idxZ = find(squeeze(any(any(track_mask, 1), 2)));
    idx_length = numel(idxZ);
    track.diameter = 2*r_track + 1;
    track.length = idx_length;
    track.mean = mean(data(track_mask(:)));
    cut = fix(idx_length/2);
    m1 = track_mask; m1(:,:,idxZ(cut+1:end))=false;
    m2 = track_mask; m2(:,:,idxZ(1:cut))=false;
    track.mean2 = [mean(data(m1(:))), mean(data(m2(:)))];
    cut = fix(idx_length/3);
    cut1 = fix(2*idx_length/3);
    m1 = track_mask; m1(:,:,idxZ(cut+1:end))=false;
    m2 = track_mask; m2(:,:,idxZ([1:cut, cut1+1:end]))=false;
    m3 = track_mask; m3(:,:,idxZ(1:cut1))=false;
    track.mean3 = [mean(data(m1(:))), mean(data(m2(:))), mean(data(m3(:)))];
    cut = fix(idx_length/4);
    cut1 = fix(idx_length/2);
    cut2 = fix(3*idx_length/4);
    m1 = track_mask; m1(:,:,idxZ(cut+1:end))=false;
    m2 = track_mask; m2(:,:,idxZ([1:cut, cut1+1:end]))=false;
    m3 = track_mask; m3(:,:,idxZ([1:cut1, cut2+1:end]))=false;
    m4 = track_mask; m4(:,:,idxZ(1:cut2))=false;
    track.mean4 = [mean(data(m1(:))), mean(data(m2(:))), mean(data(m3(:))), mean(data(m4(:)))];
  end
  disp(track)
  save('track')
  disp('');
end

% if handles.plot4 == 6
%   planes = handles.projection(1:3);
%   draw4_planes = ImageInZAxis(planes, last_hit_axis3D);
%   yx_mask_add = {};
%   zx_mask_add = {};
%   yz_mask_add = {};
%   
%   r_track = 1;
%   tracks = handles.tracks;
%   tracks{end+1} = struct('planes', draw4_planes, 'hit_ax', last_hit_axis3D);
%   for ii=1:length(tracks)
%     track_mask = false(size(data));
%     track_mask(tracks{ii}.planes(1)+r_track*(-1:1),tracks{ii}.planes(2)+r_track*(-1:1),:) =true;
%     track_mask = ImageInNormalAxis(track_mask & SelMask, tracks{ii}.hit_ax);
%     [yx_mask_add{end+1}.Mask, zx_mask_add{end+1}.Mask, yz_mask_add{end+1}.Mask]=...
%       epr_getslice3D(track_mask, planes([2,1,3]), idx_roi.x, idx_roi.y, idx_roi.z);
%     lw = 2; yx_mask_add{end}.LineWidth = lw;  zx_mask_add{end}.LineWidth = lw;  yz_mask_add{end}.LineWidth = lw;
%     cl = 'm'; yx_mask_add{end}.Color = cl;  zx_mask_add{end}.Color = cl;  yz_mask_add{end}.Color = cl;
%   end
% end


% --------------------------------------------------------------------
function mFileLoadVar_Callback(hObject, eventdata, handles) 

vars = evalin('base','whos');

vlist = {};
for ii=1:length(vars)
  if numel(find(vars(ii).size > 1)) >= 3
    vlist{end+1} = vars(ii).name;
  elseif strcmp(vars(ii).class, 'struct') && prod(vars(ii).size)
    strvars = evalin('base', ['fieldnames(',vars(ii).name,')']);
    for jj=1:length(strvars)
      elm = evalin('base', [vars(ii).name, '.', strvars{jj}]);
      if ndims(elm) >= 3
        vlist{end+1} = [vars(ii).name, '.', strvars{jj}];
      end
    end
  end
end

if ~isempty(vlist)
  [sel, ok] = listdlg('ListString', vlist);
  if ok
    the_image.(genvarname(vlist{sel})) = evalin('base', vlist{sel});
    handles = Data_Load(the_image, handles, false, true);
    UpdateMaskList(handles);
    AfterProjectionChanged(handles);
  end
else
  disp('No variables found.');
end

% --------------------------------------------------------------------
function mFileExportSlice_Callback(hObject, eventdata, handles) 
handles.export_slice = true;
handles.export = [];
try
  Draw_Image(handles, handles.projection);
catch err
  handles.export_slice = false;
  error_catcher('mFileExportSlice_Callback', err)
  return;
end

handles = guidata(handles.figure1);
assignin('base', 'Slice', handles.export);
disp('Data of displayed slice are exported to workspace as ''Slice''.');
evalin('base', 'disp(Slice)');

handles.export = [];
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function mOptionsColormap_Callback(hObject, eventdata, handles) %#ok<INUSD,DEFNU>
% Way around non-modality of colormapeditor
colormapeditor(handles.axes_1);
cmap = colormap(handles.axes_1);
%colormap(handles.axes_1, cmap);
colormap(handles.axes_2, cmap);
colormap(handles.axes_3, cmap);
colormap(handles.axScale, cmap);
% AfterProjectionChanged(handles);


% --------------------------------------------------------------------
function mOptionsGuides_Callback(hObject, eventdata, handles) 
hh = [handles.mOptGuideNo, handles.mOptGuideBlack, handles.mOptGuideGreen];
set(hh, 'Checked', 'off'); set(hObject, 'Checked', 'on');
switch hObject
  case handles.mOptGuideNo, handles.options.guides = 'no'; handles.options.font_color = 'k';
  case handles.mOptGuideBlack,  handles.options.guides = 'k:'; handles.options.font_color = 'k';
  case handles.mOptGuideGreen,  handles.options.guides = 'g:'; handles.options.font_color = 'g';
end
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
function mAddSpRes_Callback(hObject, eventdata, handles) 

image_number = get(handles.pmImageType, 'Value');
data = Data_Get(handles.figure1, image_number);

pos = handles.projection(4);

useSetUp = 0;
if isfield(handles, 'SpatResSetUp')
  button = questdlg('Do you want to reuse old setup ?', 'Spatial Resolution'); 
  if strcmp(button, 'Yes'), useSetUp = 1; end
  if strcmp(button, 'Cancel'), return; end
end

if useSetUp
  [fwhm] = SpatialResolution(squeeze(data(:,:,:,pos)), handles.Image.Size*10, handles.SpatResSetUp);
else
  [fwhm, handles.SpatResSetUp] = SpatialResolution(squeeze(data(:,:,:,pos)), mean(handles.Image.Size)*10);
  guidata(handles.figure1, handles);
end

% --------------------------------------------------------------------
function mMeasureResolution_Callback(hObject, eventdata, handles) 
image_number = get(handles.pmImageType, 'Value');
data = Data_Get(handles.figure1, image_number);
pos = handles.projection(4);
SpatialResolutionGUI(squeeze(data(:,:,:,pos)), handles.ImageSize(1)*10);

% --------------------------------------------------------------------
function mSaveSpRes_Callback(hObject, eventdata, handles) 
if isfield(handles, 'SpatResSetUp')
  SpatResSetUp = handles.SpatResSetUp;
  SpatResSetUp.file_type = 'SpatResSetUp_v1.0';
  [filename, pathname] = uiputfile( ...
    {'*.mat', 'All MATLAB Files (*.mat)'; ...
    '*.*', 'All Files (*.*)'}, 'Save as');
  if ~(isequal(filename,0) || isequal(pathname,0))
    savefile = fullfile(pathname, filename);
    save(savefile, '-struct', 'SpatResSetUp');
    disp(['File ', savefile, ' is saved.']);
  end
end

% --------------------------------------------------------------------
function mLoadSpRes_Callback(hObject, eventdata, handles) 
  [filename, pathname] = uigetfile( ...
    {'*.mat', 'All MATLAB Files (*.mat)'; ...
    '*.*', 'All Files (*.*)'}, 'Save as');
  if ~(isequal(filename,0) || isequal(pathname,0))
    loadfile = fullfile(pathname, filename);
    s1 = load(loadfile);
    disp(['File ', loadfile, ' is loaded.']);
    handles.SpatResSetUp = s1;
    guidata(handles.figure1, handles);
  end

% --------------------------------------------------------------------
function mFileTimeline_Callback(hObject, eventdata, handles) 
[handles.Timeline,isOk] = LoadImageListGUI(safeget(handles, 'Timeline', {}), safeget(handles.options, 'PathName', ''));
if isOk
  guidata(handles.figure1, handles);
  
  load_options = [];
  pars.CW  = handles.options.CWload(handles.options.CWidx(:));
  pars.ESE = handles.options.ESEload(handles.options.ESEidx(:));
  
  data_fields = {};
  Time0 = now;
  for ii=1:length(handles.Timeline)
    if ii==1
      Image1 = epr_LoadMATFile(handles.Timeline{ii}.FileName, [], pars);
      load_options = LoadCalibrationDLG(Image1.pO2_info);
%       load_options = Image1.pO2_info;
      
      % Analyse useful fields
      fields = fieldnames(Image1);
      for jj=1:length(fields)
        if isstruct(Image1.(fields{jj})), continue; end
        if strcmp(fields{jj}, 'Mask'), continue; end
        if strcmp(fields{jj}, 'Masks'), continue; end
        if ndims(Image1.(fields{jj})) ~= 3, continue; end
        % Turn matrix into double
        data_fields{end+1} = fields{jj}; %#ok<AGROW>
      end
      
      if isfield(Image1, 'StartTime')
          Image1.StartTime = datenum(Image1.StartTime);
          Image1.FinishTime = datenum(Image1.FinishTime);
      else
          Image1.StartTime = Time0 + ii*60;
          Image1.FinishTime = Image1.StartTime + 0.5;
      end
    else
      Image2 = epr_LoadMATFile(handles.Timeline{ii}.FileName, load_options, pars);
      filler = -100;
      for jj=1:length(data_fields)
        ppp = Image2.(data_fields{jj});
        ppp(~Image2.Mask) = filler;
        Image1.(data_fields{jj})(:,:,:,ii) = ppp;
      end
      if isfield(Image2, 'StartTime')
          Image1.StartTime(ii) = datenum(Image2.StartTime);
          Image1.FinishTime(ii) = datenum(Image2.FinishTime);
      else
          Image1.StartTime(ii) = Time0 + ii*60;
          Image1.FinishTime(ii) = Image1.StartTime(ii) + 0.5;
      end
    end
  end
  
  Data_Load(Image1, handles, false, false);
  handles = guidata(handles.figure1);
  Mask_Check(handles.figure1, handles.ImageDim);
  UpdateMaskList(handles);
  AfterProjectionChanged(handles);
end

% --------------------------------------------------------------------
function mAddOnCMapCreate_Callback(hObject, eventdata, handles) 
cMask = classMaskStorage(handles.figure1, 'Masks', handles.options);
image_number = get(handles.pmImageType, 'Value');
data = Data_Get(handles.figure1, image_number);

sz = size(data);

Mask = cMask.SafeGet(1, sz(1:3));
idx = find(Mask);

Excitation = zeros(size(data, 4), 1);
Excitation(9:20) = 1;
Excitation(31:39) = 1;
% Excitation = safeget(handles.Image, 'Excitation', []);
Excitation = Excitation - mean(Excitation);

if ~isempty(Excitation)
  %find correlation for each voxel
  h = waitbar(0,'Calculating Correlation Map...');
  corrmap = zeros(sz(1:3));
  for p = 1:length(idx)
    [ii,jj,kk] = ind2sub(sz(1:3), idx(p));
    temporalvoxel = squeeze(data(ii,jj,kk,:));
    %       corrmap(ii,jj,kk) = corr2(temporalvoxel,Excitation(:));
    corrmap(ii,jj,kk) = sum(temporalvoxel.*Excitation) / mean(temporalvoxel);
    waitbar(p/length(idx));
  end
  delete(h)
  %     [FileName, PathName] = uiputfile('.mat','Save Correlation Map');
  %     if isequal(FileName,0) || isequal(PathName,0)
  %         disp('User selected Cancel')
  %     else
  %         disp(['User selected ',fullfile(PathName,FileName)])
  %     end
  %     save(fullfile(PathName,FileName),'corrmap')
else
  helpdlg('No Excitation Function Specified')
end
ibGUI(corrmap)

% --------------------------------------------------------------------
function figure1_WindowScrollWheelFcn(hObject, eventdata, handles) 
coord = get(0,'PointerLocation');
ax = [handles.axes_1, handles.axes_2, handles.axes_3];
change = [3,2,1];
fpos = get(handles.figure1, 'Position');
coord = coord - fpos(1:2);
idx = handles.options.axis_idx;

for ii=1:3
  ax_coord = get(ax(ii), 'Position');
  if coord(1) >= ax_coord(1) && coord(1) <= ax_coord(1)+ax_coord(3) && ...
      coord(2) >= ax_coord(2) && coord(2) <= ax_coord(2)+ax_coord(4)
    val = eventdata.VerticalScrollCount;
        handles.projection(idx{change(ii)}) = val+handles.projection(idx{change(ii)});
    AfterProjectionChanged(handles);
    break;
  end
end

% --------------------------------------------------------------------
function mHelpAbout_Callback(~, ~, ~) 
disp('Version 2.0 alpha');

% --------------------------------------------------------------------
function mMeasureStatistics_Callback(hObject, eventdata, handles) 

image_number = get(handles.pmImageType, 'Value');
[data, data_info] = Data_Get(handles.figure1, image_number);

Masks = getappdata(handles.figure1, 'Masks');

if handles.ROIrad ~= 0
  Masks{end+1}.Mask = ...
    epr_GetSphericMask(data_info.image_dim, handles.projection, handles.ROIrad);
  Masks{end}.Name = 'ROI';
end

nPars = 8;
stat_data     = zeros(nPars, length(Masks));

stat_lheader  = cell(nPars, 1);
the_header = '';
for ii=1:length(Masks)
  the_header = [the_header, sprintf('%12s', Masks{ii}.Name)];
  data_for_statistics = data(find(Masks{ii}.Mask==1 & data~= -100));
  stat_lheader{1}='Numel in mask'; stat_data(1, ii) = numel(find(Masks{ii}.Mask==1));
  stat_lheader{2}='Numel in mask and image'; stat_data(2, ii) = numel(data_for_statistics);
  stat_lheader{3}='mean'; stat_data(3, ii) = mean(data_for_statistics(:));
  stat_lheader{4}='median'; stat_data(4, ii) = median(data_for_statistics(:));
  stat_lheader{5}='std'; stat_data(5, ii) = std(data_for_statistics(:));
  

  stat_lheader{5}='HF2.5'; stat_data(5, ii) = numel(find(data_for_statistics <= 2.5))/numel(data_for_statistics);
  stat_lheader{6}='HF5'; stat_data(6, ii) = numel(find(data_for_statistics <= 5))/numel(data_for_statistics);
  stat_lheader{7}='HF10'; stat_data(7, ii) = numel(find(data_for_statistics <= 10))/numel(data_for_statistics);
  stat_lheader{8}='HF15'; stat_data(8, ii) = numel(find(data_for_statistics <= 15))/numel(data_for_statistics);
end  
 
  
  
%   for jj=1:7
%     stat_lheader{jj-1+9}=['Connect R=', num2str(jj)]; stat_data(jj-1+9, ii) = epr_connectivity(Masks{ii}.Mask, jj);
%   end
% end

nPars = size(stat_data, 1);

fprintf('\n\n%s\n', get(handles.figure1, 'Name'));
fprintf('%-16s  %s\n', 'Image statistics', the_header);
for ii=1:nPars
  fprintf('%-16s: %s\n', stat_lheader{ii}, sprintf('%12.3f', stat_data(ii,:)));
end

return;

% [Mask, SelMask] = GetImageMask(handles, 1);
% ROImask = GetExternalMask(handles,1);
% 
% % Whole image 
% mmax = handles.options.po2.scale_minmax;
% zip_zap_mask = handles.Image.pO2 >= mmax(1) & handles.Image.pO2 <= mmax(2);
% SelMask = SelMask & zip_zap_mask;
% pO2 = handles.Image.pO2(SelMask(:));
% n = numel(pO2);
% fprintf('\n');
% disp('Image statistics:');
% fprintf('  pO2: %d pixels\n', n);
% fprintf('  pO2:  mean value %4.1f (+-%4.1f) torr\n', mean(pO2), std(pO2)/sqrt(n));
% fprintf('  pO2:  median value %4.1f torr\n', median(pO2));
% fprintf('  pO2:  standard deviation %4.1f torr\n', std(pO2));
% fprintf('  pO2:  HF5  [0-1] %4.2f\n', numel(find(pO2 <= 5))/numel(find(pO2 > 5)));
% fprintf('  pO2:  HF10 [0-1] %4.2f\n', numel(find(pO2 <= 10))/numel(find(pO2 > 10)));
% 
% % ROI
% if ~isempty(ROImask)
%   ROImask = ROImask & SelMask;
%   disp('ROI statistics:');
%   pO2 = handles.Image.pO2(ROImask(:));
%   n = numel(pO2);
%   fprintf('  pO2: %d pixels\n', n);
%   fprintf('  pO2:  mean value %4.1f (+-%4.1f) torr\n', mean(pO2), std(pO2)/sqrt(n));
%   fprintf('  pO2:  median value %4.1f torr\n', median(pO2));
%   fprintf('  pO2:  standard deviation %4.1f torr\n', std(pO2));
%   fprintf('  pO2:  HF5  [0-1] %4.2f\n', numel(find(pO2 <= 5))/numel(find(pO2 > 5)));
%   fprintf('  pO2:  HF10 [0-1] %4.2f\n', numel(find(pO2 <= 10))/numel(find(pO2 > 10)));
%   
%   ElseMask = SelMask & (~ROImask);
%   disp('Except ROI statistics:');
%   pO2 = handles.Image.pO2(ElseMask(:));
%   n = numel(pO2);
%   fprintf('  pO2: %d pixels\n', n);
%   fprintf('  pO2:  mean value %4.1f (+-%4.1f) torr\n', mean(pO2), std(pO2)/sqrt(n));
%   fprintf('  pO2:  median value %4.1f torr\n', median(pO2));
%   fprintf('  pO2:  standard deviation %4.1f torr\n', std(pO2));
%   fprintf('  pO2:  HF5  [0-1] %4.2f\n', numel(find(pO2 <= 5))/numel(find(pO2 > 5)));
%   fprintf('  pO2:  HF10 [0-1] %4.2f\n', numel(find(pO2 <= 10))/numel(find(pO2 > 10)));  
% end 
% 
% 
% end

% --------------------------------------------------------------------
function mROIfit_Callback(hObject, ~, handles)

if isempty(handles.Image.Raw) 
    warning('ibGUI::mROIfit_Callback', 'RAW data are not available.');
    return; 
end

szRaw = size(handles.Image.Raw);
sz = szRaw(1:3);

if isempty(handles.Image.Amp) 
    handles.Image.Amp = zeros(sz);  handles.Image.pO2 = zeros(sz);  handles.Image.Mask = false(sz);
    
    % generate selection list
    str = {};
    handles.Image_Idx = zeros(length(handles.options.struct_def), 1);
    for ii=1:length(handles.options.struct_def)
      if isfield(handles.Image, handles.options.field_def{ii}) && ~isempty(handles.Image.(handles.options.field_def{ii}))
        str{end+1} = ['Slices: ',handles.options.field_def{ii}];
        handles.Image_Idx(length(str)) = ii;
      end
    end
    set(handles.pmImageType, 'String', str, 'Value', 1);

end

ROImask = epr_GetSphericMask(sz, handles.projection, handles.ROIrad);
[Mask, SelMask] = GetImageMask(handles, 1);

if hObject == handles.mROIfit
    FITmask = ROImask & (~SelMask);
else
    FITmask = ROImask;
end
if isempty(find(FITmask, 1)), return; end

TT   = safeget(handles.Image.raw_info, 'T1', [])*1E6;
tau2 = 2*safeget(handles.Image.raw_info, 'tau', [])*1E6;

number_of_echoes = szRaw(4);
yyy = reshape(handles.Image.Raw,[sz(1)*sz(2)*sz(3), number_of_echoes]);
idx = find(FITmask(:))';
fprintf('Fitting %i voxels.\n', numel(idx));
fit_y = double(yyy(idx,:));

[fit_amp, fit_t1, fit_inv, fit_t2, fit_err_mask, fit_data.Perr] = fit_recovery_simultaneous(fit_y, tau2, TT);

tmp = handles.Image.Amp(:);
tmp(FITmask) = fit_amp / handles.Image.pO2_info.amp1mM;
handles.Image.Amp = reshape(tmp, sz);
tmp = handles.Image.pO2(:);
tmp(FITmask) = epr_T2_PO2(fit_t1, fit_amp, true(size(idx)), handles.Image.pO2_info);
handles.Image.pO2 = reshape(tmp, sz);
tmp = handles.Image.Mask(:);
tmp(FITmask) = true;
handles.Image.Mask = reshape(tmp, sz);
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
function mFileRandom_Callback(hObject, ~, handles) 

answer=inputdlg({'Dimension 1'; 'Dimension 2'; 'Dimension 3'},...
  'Matrix parameters',1, {'64'; '64'; '64'});
dim1 = fix(str2double(answer{1}));
dim2 = fix(str2double(answer{2}));
dim3 = fix(str2double(answer{3}));
if ~isempty(answer)

  new_data.Random = rand(dim1, dim2, dim3);
  handles = Data_Load(new_data, handles, false, true);
  
  AfterProjectionChanged(handles);
end

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------- I M A G E     M A N A G M E N T  -------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
function [data, data_info] = Data_Get(hGUI, im_number)
Images = getappdata(hGUI, 'Images');
if im_number > length(Images), data = []; data_info= [];  return; end
data = Images{im_number}.Image;
data_info.image_type = Images{im_number}.Type;
data_info.image_dim = Images{im_number}.Dim;
data_info.StartTime = safeget(Images{im_number}, 'StartTime', now);
data_info.FinishTime = safeget(Images{im_number}, 'FinishTime', now);

% --------------------------------------------------------------------
function [data, image_type, image_dim] = Data_Get3D(hGUI, im_number, slice4D)
Images = getappdata(hGUI, 'Images');
if im_number > length(Images), data = []; image_type= []; image_dim = []; return; end
data = Images{im_number}.Image;
if size(data,4) ~= 1, data = data(:,:,:,slice4D); end
image_type = Images{im_number}.Type;
image_dim = Images{im_number}.Dim;

% --------------------------------------------------------------------
function [image_type, image_dim] = Data_Stat(hGUI, im_number)
Images = getappdata(hGUI, 'Images');
if im_number > length(Images), image_type= []; image_dim = []; return; end
image_type = Images{im_number}.Type;
image_dim = Images{im_number}.Dim;

% --------------------------------------------------------------------
function [slice, nslice] = Data_GetSlice(hGUI, im_number, hit_axis, proj)
% get slice of image for the given axis
Images = getappdata(hGUI, 'Images');
if im_number > length(Images), slice= []; nslice = []; return; end
data = Images{im_number}.Image;
switch hit_axis
  case 1, slice = squeeze(data(:,:,proj(3),proj(4))); nslice = proj(3);
  case 2, slice = squeeze(data(:,proj(2),:,proj(4))); nslice = proj(2);
  case 3, slice = squeeze(data(proj(1),:,:,proj(4)))';  nslice = proj(1);
end

% --------------------------------------------------------------------
function [data, image_type, image_dim] = Data_GetXYZ(hGUI, im_number, ijk)
Images = getappdata(hGUI, 'Images');
if im_number > length(Images), data = []; image_type= []; image_dim = []; return; end
% slice4D = 1;
% if size(data,4) ~= 1, data = data(:,:,:,slice4D); end
image_type = Images{im_number}.Type;
image_dim = Images{im_number}.Dim;
for ii=1:3, ijk(ii) = min(max(ijk(ii), 1), image_dim(ii)); end
data = Images{im_number}.Image(ijk(1), ijk(2), ijk(3));

% --------------------------------------------------------------------
function Image = Data_ToZ(Image, hit_ax)
switch hit_ax
  case 2, Image = permute(Image, [1,3,2]);
  case 3, Image = permute(Image, [3,2,1]);
end

% --------------------------------------------------------------------
function Image = Data_FromZ(Image, hit_ax)
switch hit_ax
  case 2, Image = permute(Image, [1,3,2]);
  case 3, Image = permute(Image, [3,2,1]);
end
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------- M A S K     U S E R    I N T E R F A C E -----------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
function mViewMaskToolbox_Callback(hObject, ~, handles) 
if isempty(handles.Toolbars.hMaskToolbar)
  handles.Toolbars.hMaskToolbar = uitoolbar();
  % handles.Toolbars.hMaskToolbar2 = uitoolbar();
  set(handles.mViewMaskToolbox, 'Checked', 'on');
  icons = load('ibToolBoxMask.mat');

  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.LoadFile, 'TooltipString', 'Load mask', ...
    'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''Mask_Load'',guidata(gcbo))','HandleVisibility','off');
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.LoadFile, 'TooltipString', 'Load mask from workspace', ...
    'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''Mask_LoadWsps'',guidata(gcbo))','HandleVisibility','off');
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.SaveFile, 'TooltipString', 'Save mask', ...
    'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''Mask_Save'',guidata(gcbo))','HandleVisibility','off'); 
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.SaveFileRes, 'TooltipString', 'Save mask in given resolution', ...
    'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''Mask_Save_Res'',guidata(gcbo))','HandleVisibility','off'); 
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.SaveFileExcel, 'TooltipString', 'Save data in mask to excel', ...
    'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''Mask_Save_Excel'',guidata(gcbo))','HandleVisibility','off'); 

  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.Undo, 'TooltipString', 'Undo', ...
    'Separator', 'on', 'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''Mask_Undo'',guidata(gcbo))','HandleVisibility','off');
%   uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.Redo, 'TooltipString', 'Redo', ...
%     'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''Mask_Redo'',guidata(gcbo))','HandleVisibility','off'); 
  
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.AddPoly, 'TooltipString', 'AddPoly', ...
    'Separator', 'on', 'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''AddPoly'',guidata(gcbo))','HandleVisibility','off');
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.AddFreeHand, 'TooltipString', 'AddFreeHand', 'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''AddFreeHand'',guidata(gcbo))','HandleVisibility','off');
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.SetAll, 'TooltipString', 'Set all', ...
    'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''SetAll'',guidata(gcbo))','HandleVisibility','off');
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.RemovePoly, 'TooltipString', 'ErasePoly', 'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''ErasePoly'',guidata(gcbo))','HandleVisibility','off');
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.EraseFreeHand, 'TooltipString', 'EraseFreeHand', 'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''EraseFreeHand'',guidata(gcbo))','HandleVisibility','off'); 
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.EraseAll, 'TooltipString', 'Erase all', ...
    'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''EraseAll'',guidata(gcbo))','HandleVisibility','off');
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.SelectByThreshold, 'TooltipString', 'SelectByThreshold', ...
    'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''SelectByThreshold'',guidata(gcbo))','HandleVisibility','off');
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.Otsu, 'TooltipString', 'Otsu segmentation', ...
    'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''Otsu'',guidata(gcbo))','HandleVisibility','off');
  
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.SetAll3D, 'TooltipString', 'Set all (3D)', ...
    'Separator', 'on', 'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''SetAll3D'',guidata(gcbo))','HandleVisibility','off');
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.EraseAll3D, 'TooltipString', 'Erase all (3D)', ...
    'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''EraseAll3D'',guidata(gcbo))','HandleVisibility','off');
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.SelectByThreshold3D, 'TooltipString', 'SelectByThreshold (3D)', ...
    'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''SelectByThreshold3D'',guidata(gcbo))','HandleVisibility','off');
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.Interpolate3D, 'TooltipString', 'Interpolate (3D)', ...
    'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''Interpolate'',guidata(gcbo))','HandleVisibility','off');
  
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.CloseImage3D, 'TooltipString', 'Close image (3D)', ...
    'Separator', 'on', 'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''CloseImage3D'',guidata(gcbo))','HandleVisibility','off');
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.OpenImage3D, 'TooltipString', 'Open image (3D)', ...
    'Separator', 'off', 'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''OpenImage3D'',guidata(gcbo))','HandleVisibility','off');
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.RemoveSpecles, 'TooltipString', 'Remove specles (3D)', ...
    'Separator', 'off', 'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''RemoveSpecles'',guidata(gcbo))','HandleVisibility','off');
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.FillHoles3D, 'TooltipString', 'Fill holes (3D)', ...
    'Separator', 'off', 'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''FillHoles3D'',guidata(gcbo))','HandleVisibility','off');
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.Otsu3D, 'TooltipString', 'Otsu segmentation (3D)', ...
    'ClickedCallback', 'ibGUI(''MaskToolBar_Callback'',gcbo,''Otsu3D'',guidata(gcbo))','HandleVisibility','off');

  %   uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.FloodFill, 'TooltipString', 'FloodFill', 'Separator', 'on');
  %   uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.CircleBrush, 'TooltipString', 'CircleBrush');
  %   uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.CircleEraser, 'TooltipString', 'CircleEraser');
  %
  %   uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.CloseImage, 'TooltipString', 'CloseImage');
  %   uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.OpenImage, 'TooltipString', 'OpenImage');
  %   uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.Propogate, 'TooltipString', 'Propogate');
  %   uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.DeleteContour, 'TooltipString', 'DeleteContour');
  
  uipushtool(handles.Toolbars.hMaskToolbar, 'CData', icons.Close, 'TooltipString', 'Close', ...
    'Separator', 'on', 'ClickedCallback', 'ibGUI(''mViewMaskToolbox_Callback'',gcbo,''Close'',guidata(gcbo))','HandleVisibility','off');
%   uipushtool(handles.Toolbars.hMaskToolbar2, 'CData', icons.Close, 'TooltipString', 'Close', ...
%     'Separator', 'on', 'ClickedCallback', 'ibGUI(''mViewMaskToolbox_Callback'',gcbo,''Close'',guidata(gcbo))','HandleVisibility','off');
  
  set(handles.pMaskControl, 'Visible', 'on');
else
  delete(handles.Toolbars.hMaskToolbar );
  handles.Toolbars.hMaskToolbar = [];
  %delete(handles.Toolbars.hMaskToolbar2 );
  handles.Toolbars.hMaskToolbar2 = [];
  set(handles.mViewMaskToolbox, 'Checked', 'off');
  set(handles.pMaskControl, 'Visible', 'off');
end
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
function MaskToolBar_Callback(hObject, fun_call, handles) 
if ~isempty(handles.op_mode), return; end
cMask = classMaskStorage(handles.figure1, 'Masks', handles.options);

switch fun_call
  case 'AddPoly'
    disp('Click on an axis to start conturing');
    handles.op_mode = 'AddPoly';
    guidata(handles.figure1, handles);
  case 'ErasePoly'
    disp('Click on an axis to start conturing');
    handles.op_mode = 'ErasePoly';
    guidata(handles.figure1, handles);
  case 'AddFreeHand'
    disp('Click on an axis to start conturing');
    handles.op_mode = 'AddFreeHand';
    guidata(handles.figure1, handles);
  case 'EraseFreeHand'
    disp('Click on an axis to start conturing');
    handles.op_mode = 'EraseFreeHand';
    guidata(handles.figure1, handles);
  case 'SelectByThreshold'
    disp('Click on an axis to apply');
    handles.op_mode = 'SelectByThreshold';
    guidata(handles.figure1, handles);
  case 'SetAll'
    disp('Click on an axis to apply');
    handles.op_mode = 'SetAll';
    guidata(handles.figure1, handles);
  case 'EraseAll'
    disp('Click on an axis to apply');
    handles.op_mode = 'EraseAll';
    guidata(handles.figure1, handles);
  case 'Mask_Load'
    dirs     = safeget(handles.ini, 'Directories', []);
    old_path = safeget(dirs, 'SourcePath', 'C:/');
    
    [FileName,PathName] = uigetfile({'*.mat', 'Matlab files (*.mat)'},'Load file', old_path);
    fname = fullfile(PathName, FileName);
    
    if FileName ~=0
      new_mask = Mask_Load(fname);
      mask_idx = get(handles.lbMaskList, 'Value');
      Mask_Replace(handles.figure1, new_mask, mask_idx, handles.ImageDim);
      AfterProjectionChanged(handles);
    end
  case 'Mask_LoadWsps'
    image_number = get(handles.pmImageType, 'Value');
    Data = Data_Get3D(handles.figure1, image_number, handles.projection(4));
    sz = size(Data);

    vars = evalin('base','whos');
    vlist = {};
      for ii=1:length(vars)
        if isequal(vars(ii).size,sz)
          vlist{end+1} = vars(ii).name;
        end
      end
      
      if ~isempty(vlist)
        vlist = sort(vlist);
        [sel, ok] = listdlg('ListString', sort(vlist));
        if ok
          new_mask = evalin('base', vlist{sel});
          mask_idx = get(handles.lbMaskList, 'Value');
          Mask_Replace(handles.figure1, new_mask, mask_idx, handles.ImageDim);
          AfterProjectionChanged(handles);
        end
      else
        disp('No masks found in the workspace.');
      end
  case 'Mask_Save'
    dirs     = safeget(handles.ini, 'Directories', []);
    old_path = safeget(dirs, 'SourcePath', 'C:/');
         
    [FileName,PathName] = uiputfile({'*.mat', 'Matlab files (*.mat)'},'Save file', old_path);
    fname = fullfile(PathName, FileName);
    
    if FileName ~=0
      mask_idx = get(handles.lbMaskList, 'Value');
      Mask_Save(handles.figure1, mask_idx, fname);
    end    
  case 'Mask_Save_Res'
    dirs     = safeget(handles.ini, 'Directories', []);
    old_path = safeget(dirs, 'SourcePath', 'C:/');
    
    res = inputdlg('Matrix resolution','Martix',1, {'64'});
    if ~isempty(res)
      new_resolution = str2double(res);
    end

    [FileName,PathName] = uiputfile({'*.mat', 'Matlab files (*.mat)'},'Save file', old_path);
    fname = fullfile(PathName, FileName);
    
    if FileName ~=0
      mask_idx = get(handles.lbMaskList, 'Value');
      Mask_Save_Res(handles.figure1, mask_idx, fname, new_resolution);
    end
  case 'Mask_Save_Excel'
    dirs     = safeget(handles.ini, 'Directories', []);
    old_path = safeget(dirs, 'SourcePath', 'C:/');
    
    [FileName,PathName,filterindex] = uiputfile({'*.xls', 'Excel files (*.xls)';...
                                     '*.csv', 'Comma separated (*.csv)'},'Save file', old_path);
    fname = fullfile(PathName, FileName);
    
    if FileName ~=0
      mask_idx = get(handles.lbMaskList, 'Value');
      image_number = get(handles.pmImageType, 'Value');
      mydata = Data_Get(handles.figure1, image_number);
      mymask = cMask.Get(mask_idx);
      
      switch filterindex
        case 1
            xlswrite(fname, mydata(mymask(:)));
        case 2
            f = mydata(mymask(:));
            save(fname, 'f','-ascii');
      end
    end
  case 'Mask_Undo'
    cMask.Restore();
    AfterProjectionChanged(handles);
  case 'Interpolate'
    disp('Click on an axis to apply');
    handles.op_mode = 'Interpolate';
    guidata(handles.figure1, handles);
  case 'SetAll3D'
    mask_idx = get(handles.lbMaskList, 'Value');
    cMask.Set(mask_idx, true(handles.ImageDim(1:3)));
    AfterProjectionChanged(handles);
  case 'EraseAll3D'
    mask_idx = get(handles.lbMaskList, 'Value');
    cMask.Set(mask_idx, false(handles.ImageDim(1:3)));
    AfterProjectionChanged(handles);
  case 'SelectByThreshold3D'
    image_number = get(handles.pmImageType, 'Value');
    Data = Data_Get3D(handles.figure1, image_number, handles.projection(4));
    mask_idx = get(handles.lbMaskList, 'Value');
    if mask_idx == 1 % data mask
        Mask = true(size(Data));
    else
        Mask = cMask.SafeGet(1, handles.ImageDim);
    end
    answer=inputdlg({'Lower threshold'; 'Higher threshold'},...
      'Mask parameters',1, {num2str(min(Data(Mask(:)))); num2str(max(Data(Mask(:))))});
    if ~isempty(answer)
      min_thres = str2double(answer{1});
      max_thres = str2double(answer{2});
      cMask.Set(mask_idx, Data >= min_thres & Data <= max_thres & Mask);
      AfterProjectionChanged(handles);
    end
  case 'Otsu'
    disp('Click on an axis to apply');
    handles.op_mode = 'Otsu';
    guidata(handles.figure1, handles);
  case 'Otsu3D'
    image_number = get(handles.pmImageType, 'Value');
    Data = Data_Get3D(handles.figure1, image_number, handles.projection(4));
    Data = Data / max(Data(:));
    answer=inputdlg({'Number of thresholds'; 'Use range'},...
      'Otsu algorithm parameters',1, {'4';'4'});
    if ~isempty(answer)
      n_threshold = fix(str2double(answer{1}))-1;
      use_threshold = fix(str2double(answer{2}));
      mask_idx = get(handles.lbMaskList, 'Value');
  
      Levels = multithresh(Data,n_threshold);
      NewMask = imquantize(Data,Levels);
%      ibGUI(NewMask);
      cMask.Set(mask_idx, NewMask == use_threshold);
      AfterProjectionChanged(handles);        
    end
  case 'RemoveSpecles'
    mask_idx = get(handles.lbMaskList, 'Value');
    NewMask = cMask.Get(mask_idx);
    
    for ii=1:size(NewMask,3)
      current_mask = NewMask(:,:,ii);
      NewMask(:,:,ii) = epr_AutoMask2D('Largest', [], current_mask, []);
    end

    cMask.Set(mask_idx, NewMask);
    AfterProjectionChanged(handles);
  case 'FillHoles3D'
    mask_idx = get(handles.lbMaskList, 'Value');
    NewMask = cMask.Get(mask_idx);

    for ii=1:size(NewMask,3)
      NewMask(:,:,ii) = imfill(NewMask(:,:,ii),'holes');
    end
    
    cMask.Set(mask_idx, NewMask);
    AfterProjectionChanged(handles);    
  case 'CloseImage3D'
    mask_idx = get(handles.lbMaskList, 'Value');
    NewMask = cMask.Get(mask_idx);
    se = epr_strel('sphere', 2);

    cMask.Set(mask_idx, imclose(NewMask, se));
    AfterProjectionChanged(handles);    
  case 'OpenImage3D'
    mask_idx = get(handles.lbMaskList, 'Value');
    NewMask = cMask.Get(mask_idx);
    se = epr_strel('sphere', 2);

    cMask.Set(mask_idx, imopen(NewMask, se));
    AfterProjectionChanged(handles);
end

% --------------------------------------------------------------------
function pbMaskButtons_Callback(hObject, eventdata, handles)

switch hObject
  case handles.pbAddMask
    % GUI call to add new mask
    str = get(handles.pmNewMask, 'String');
    pos = get(handles.pmNewMask, 'Value');
    mask_idx=Mask_Add(handles.figure1, [], str{pos});
    Mask_Check(handles.figure1, handles.ImageDim)
    UpdateMaskList(handles);
    set(handles.lbMaskList, 'Value', mask_idx);
  case handles.pbRemoveMask
    mask_idx = get(handles.lbMaskList, 'Value');
    Mask_Remove(handles.figure1, mask_idx);
    Mask_Check(handles.figure1, handles.ImageDim)
    UpdateMaskList(handles);
    AfterProjectionChanged(handles);
  case handles.pbEditMask
    Masks = getappdata(handles.figure1, 'Masks');
    mask_idx = get(handles.lbMaskList, 'Value');
    if mask_idx < 0 || mask_idx > length(Masks)
      warning('ibGUI::pbEditMask_Callback', 'Mask is not selected.');
      return;
    end

    answer=inputdlg({'Name'; 'Color [auto/b/g/r/c/m/y/k/w/-'; 'Statistics'; 'LW'},...1
      'Mask parameters',1, {Masks{mask_idx}.Name; Masks{mask_idx}.Color; sprintf('%i', safeget(Masks{mask_idx}, 'MakeStat', 1)); sprintf('%i',safeget(Masks{mask_idx}, 'LW', 1))});

    if ~isempty(answer)
      Masks{mask_idx}.Name = answer{1};
      Masks{mask_idx}.Color = answer{2};
      Masks{mask_idx}.MakeStat = str2double(answer{3});
      Masks{mask_idx}.LW = str2double(answer{4});
      setappdata(handles.figure1, 'Masks', Masks);
      Mask_Check(handles.figure1, handles.ImageDim)
      UpdateMaskList(handles);
      AfterProjectionChanged(handles);
    end
  case handles.pbShowMask
    Masks = getappdata(handles.figure1, 'Masks');
    mask_idx = get(handles.lbMaskList, 'Value');
    if mask_idx < 0 || mask_idx > length(Masks)
      warning('ibGUI::pbEditMask_Callback', 'Mask is not selected.');
      return;
    end
    if contains(Masks{mask_idx}.Color, '-')
      Masks{mask_idx}.Color = 'auto';
    else
      Masks{mask_idx}.Color = '-';
    end
    setappdata(handles.figure1, 'Masks', Masks);
    Mask_Check(handles.figure1, handles.ImageDim)
    UpdateMaskList(handles);
    AfterProjectionChanged(handles);
end

% --------------------------------------------------------------------
function pbPosToolMaskBox_Callback(hObject, eventdata, handles) 
if hObject == handles.pbPos1, handles.MaskToolBoxPos = 1;
elseif hObject == handles.pbPos2, handles.MaskToolBoxPos = 2;
elseif hObject == handles.pbPos3, handles.MaskToolBoxPos = 3;
else, handles.MaskToolBoxPos = 4;
end
guidata(hObject, handles);
figure1_ResizeFcn(hObject, eventdata, handles);

% --------------------------------------------------------------------
function lbMaskList_Callback(hObject, ~, handles) 

% --------------------------------------------------------------------
function mContoursPreserveDataMAsk_Callback(hObject, ~, handles) 
handles.options.isPreserveDataMask = ~safeget(handles.options, 'isPreserveDataMask', false);
opt = {'off', 'on'};
set(handles.mContoursPreserveDataMAsk, 'Checked', opt{handles.options.isPreserveDataMask+1});
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
function mContoursMaskRaw_Callback(hObject, ~, handles)
handles.options.isMaskRaw = ~safeget(handles.options, 'isMaskRaw', false);
opt = {'off', 'on'};
set(handles.mContoursMaskRaw, 'Checked', opt{handles.options.isMaskRaw+1});
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
function mContoursXRay_Callback(hObject, ~, handles) 
handles.options.isXRayView = ~safeget(handles.options, 'isXRayView', false);
opt = {'off', 'on'};
set(handles.mContoursXRay, 'Checked', opt{handles.options.isXRayView+1});
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
function mContoursLegend_Callback(hObject, ~, handles) 
handles.options.isShowContourLegend = ~safeget(handles.options, 'isShowContourLegend', false);
opt = {'off', 'on'};
set(handles.mContoursLegend, 'Checked', opt{handles.options.isShowContourLegend+1});
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
function mContoursRestrict_Callback(hObject, ~, handles)
handles.options.isContoursRestrict = ~safeget(handles.options, 'isContoursRestrict', false);
opt = {'off', 'on'};
set(handles.mContoursRestrict, 'Checked', opt{handles.options.isContoursRestrict+1});
AfterProjectionChanged(handles);

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------- M A S K     M A N A G M E N T  ---------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
function UpdateMaskList(handles)
% Add masks names to the selection list (lbMaskList)
Masks = getappdata(handles.figure1, 'Masks');

mask_idx = get(handles.lbMaskList, 'Value');
str = cell(length(Masks), 1);

for ii=1:length(Masks)
  str{ii} = Masks{ii}.Name;
end

set(handles.lbMaskList, 'String', str, 'Value', min(mask_idx, length(str)));

% --------------------------------------------------------------------
function new_mask_idx = Mask_Add(hGUI, new_mask, new_mask_name)
% Add new mask
Masks = getappdata(hGUI, 'Masks');
new_name_exp = regexp(new_mask_name, '(?<name>\D*)(?<number>\d*)', 'names');
try
  mask_idx = str2double(new_name_exp.number)+1;
catch err
  mask_idx = 1;
end

% Find mask with the same name
name_is_found = true;
while name_is_found
  name_is_found = false;
  for ii=1:length(Masks)
    if isequal(new_mask_name, Masks{ii}.Name)
      name_is_found = true;
      new_mask_name = [new_name_exp.name, num2str(mask_idx)];
      mask_idx = mask_idx + 1;
    end
  end
end

Masks{end+1}.Mask = new_mask;
Masks{end}.Name = new_mask_name;
Masks{end}.Color = 'auto';
new_mask_idx=length(Masks);
setappdata(hGUI, 'Masks', Masks);

% --------------------------------------------------------------------
function Mask_Replace(hGUI, mask_data, mask_name, image_dimensions)
% Add new mask
Masks = getappdata(hGUI, 'Masks');
if isnumeric(mask_name)
  mask_idx = mask_name;
else
end
if mask_idx < 0 || mask_idx > length(Masks)
  warning('ibGUI::Mask_Replace', 'Mask was not replaced. Wrong mask index.');
  return;
end
sz = size(mask_data); if length(sz) == 2, sz(3)=1; end
if ~isempty(mask_data) &&  ~isequal(sz(1:3), image_dimensions(1:3));
  warning('MATLAB:ibGUI', 'Mask was not replaced. Mask and image dimensions are different.');
  return;
end
Masks{mask_idx}.Mask = mask_data;
setappdata(hGUI, 'Masks', Masks);

% --------------------------------------------------------------------
% Verify that there are no masks with duplicate names and mask have correct
% dimensions
function Mask_Check(hGUI, image_dimensions)
Masks = getappdata(hGUI, 'Masks');

was_removed = true;
while was_removed
  was_removed = false;
  for ii=1:length(Masks)
    sz = size(Masks{ii}.Mask); if length(sz) == 2, sz(3)=1; end
    if ~isempty(Masks{ii}.Mask) &&  ~isequal(sz(1:3), image_dimensions(1:3))
      Masks(ii) = [];
      warning('ibGUI::Mask_Check', 'Mask was removed. Mask and image dimensions are different.');
      was_removed = true;
      break;
    end
  end
end

% leave only first three dimesions of the mask
for ii=1:length(Masks)
  if ~isempty(Masks{ii}.Mask)
    if ndims(Masks{ii}.Mask) == 4
      Masks{ii}.Mask = Masks{ii}.Mask(:,:,:,1);
    elseif ndims(Masks{ii}.Mask) == 2
      Masks{ii}.Mask(:,:,1) = Masks{ii}.Mask;
    end
  end
end

% Data mask should be present
if isempty(Masks)
  Masks{1}.Mask = [];
  Masks{1}.Color = 'auto';
  Masks{1}.MakeStat = true;
end

% SOme fields have to be present
for ii=1:length(Masks)
  Masks{ii}.MakeStat = safeget(Masks{ii}, 'MakeStat', true);
end

% First mask is always 'Data mask'
if isempty(Masks{1}.Mask), Masks{1}.Mask = true(image_dimensions(1:3)); end
Masks{1}.Name = 'Data mask';

setappdata(hGUI, 'Masks', Masks);

% --------------------------------------------------------------------
function Mask_Remove(hGUI, mask_name)
% Remove mask
Masks = getappdata(hGUI, 'Masks');
if isnumeric(mask_name)
  mask_idx = mask_name;
else
end
% First mask is never removed
if mask_idx >= 2 && mask_idx <= length(Masks)
  Masks(mask_name) = [];
end
setappdata(hGUI, 'Masks', Masks);

% --------------------------------------------------------------------
function slice3D = Mask_GetSlice(hGUI, mask_slice, mask_dims, midx, hit_axis, slice)
Masks = getappdata(hGUI, 'Masks');
if length(Masks) < midx, return; end

MaskToChange = Masks{midx}.Mask;
if isempty(MaskToChange), MaskToChange = false(mask_dims); end

switch hit_axis
  case 1, slice3D = reshape(mask_slice, [mask_dims(1), mask_dims(2), 1]);
  case 2, slice3D = reshape(mask_slice, [mask_dims(1), 1, mask_dims(3)]);
  case 3, slice3D = reshape(mask_slice, [1, mask_dims(2), mask_dims(3)]);
end

% --------------------------------------------------------------------
function Mask_ApplySlice(hGUI, mask_slice, mask_dims, midx, hit_axis, slice, contour_mode, options)
if midx == 1 && options.isPreserveDataMask
  warning('ibGUI::Mask_ApplySlice: Data mask was not modified. Uncheck data mask preservation.');
  return;
end
% Modify mask with the given slice 
Masks = getappdata(hGUI, 'Masks');
if length(Masks) < midx, return; end

MaskToChange = Masks{midx}.Mask;
if isempty(MaskToChange), MaskToChange = false(mask_dims); end

switch hit_axis
  case 1, slice3D = reshape(mask_slice, [mask_dims(1), mask_dims(2), 1]);
  case 2, slice3D = reshape(mask_slice, [mask_dims(1), 1, mask_dims(3)]);
  case 3, slice3D = reshape(mask_slice, [1, mask_dims(2), mask_dims(3)]);
end

switch contour_mode
  case 'replace'
    switch hit_axis
      case 1, MaskToChange(:,:, slice) = slice3D;
      case 2, MaskToChange(:, slice,:) = slice3D;
      case 3, MaskToChange(slice, :,:) = slice3D;
    end
  case 'add'
    switch hit_axis
      case 1, MaskToChange(:,:, slice) = MaskToChange(:,:, slice) | slice3D;
      case 2, MaskToChange(:, slice,:) = MaskToChange(:, slice,:) | slice3D;
      case 3, MaskToChange(slice, :,:) = MaskToChange(slice, :,:) | slice3D;
    end
  case 'erase'
    switch hit_axis
      case 1, MaskToChange(:,:, slice) = MaskToChange(:,:, slice) & (~slice3D);
      case 2, MaskToChange(:, slice,:) = MaskToChange(:, slice,:) & (~slice3D);
      case 3, MaskToChange(slice, :,:) = MaskToChange(slice, :,:) & (~slice3D);
    end
end

cMask = classMaskStorage(hGUI, 'Masks', options);
cMask.Set(midx, MaskToChange);

% --------------------------------------------------------------------
function [loaded_mask, description] = Mask_Load(fname)

loaded_mask = [];
description = [];
try
  s1 = load(fname, 'file_type');
  disp(['Mask ', fname, ' is loaded.']);
  
  switch lower(s1.file_type)
    case 'imagemask_v1.0'
      s1 = load(fname, 'Mask');
      loaded_mask = s1.Mask;
    case 'arbuzgeneric_v1.0'
      disp('arbuzgeneric_v1.0 mask format is supplied.');
      s1 = load(fname);
      loaded_mask = s1.data;
    otherwise
      warning('ibGUI::Mask_Load', 'File format is not recognized.');
      return;
  end
catch err
end

% --------------------------------------------------------------------
function Mask_Save(hGUI, mask_name, fname)
Masks = getappdata(hGUI, 'Masks');
if isnumeric(mask_name)
  mask_idx = mask_name;
else
end
if mask_idx < 0 || mask_idx > length(Masks)
  warning('ibGUI::Mask_Save: Mask was not saved. Wrong mask index.');
  return;
end

s1.file_type = 'ImageMask_v1.0';
s1.Mask = Masks{mask_name}.Mask; %#ok<STRNU>
try
  save(fname, '-struct', 's1');
  disp(['Mask ', fname, ' is saved.']);
catch err
  warning('ibGUI::Mask_Save', err);
end

% --------------------------------------------------------------------
function Mask_Save_Res(hGUI, mask_name, fname, new_res)
Masks = getappdata(hGUI, 'Masks');
if isnumeric(mask_name)
  mask_idx = mask_name;
else
end
if mask_idx < 0 || mask_idx > length(Masks)
  warning('ibGUI::Mask_Save: Mask was not saved. Wrong mask index.');
  return;
end

old_res = size(Masks{mask_name}.Mask);

s1.file_type = 'ImageMask_v1.0';
s1.Mask = reslice_volume(hmatrix_scale(new_res*[1,1,1]./old_res), eye(4), ...
    zeros(new_res*[1,1,1]), double(Masks{mask_name}.Mask), 0, 1) > 0.5; %#ok<STRNU>
try
  save(fname, '-struct', 's1');
  disp(['Mask ', fname, ' is saved.']);
catch err
  warning('ibGUI::Mask_Save', err);
end

% --------------------------------------------------------------------
function mOptReference_Callback(hObject, eventdata, handles)

pars.REFxyz = handles.REFxyz;
pars.NCxyz = handles.NCxyz;
pars.ImageSize = handles.ImageSize;
res = SetCoordinateSystemDLG(pars);
if ~isempty(handles.REFxyz) && ~isempty(handles.NCxyz)
  handles.REFxyz = res.REFxyz;
  handles.NCxyz = res.NCxyz;
  
  handles.A = GetA(handles);

  AfterProjectionChanged(handles);
end

% --------------------------------------------------------------------
function AAA = GetA(handles)

  % transform to coordinate system
  Astereo2reference = hmatrix_translate(-handles.NCxyz);
  Areference2grad = hmatrix_translate(handles.REFxyz);
  
  A = Astereo2reference*Areference2grad;
  
  FOV = handles.options.size.Size;
  if safeget(handles.options.size, 'isFOV', 0)
    FOV = handles.options.size.FOV;
  end
  pix2size = hmatrix_translate(-(handles.options.size.Dim+1)/2)*hmatrix_scale(FOV./(handles.options.size.Dim-1));
  AAA = A/pix2size;

% --------------------------------------------------------------------
function mAddOnScatterPlot_Callback(hObject, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks', handles.options);

str = get(handles.pmImageType, 'String');
[selX, isOK] = listdlg('PromptString','Select abscissa image:','ListString', str);
if ~isOK, return; end
[selY, isOK] = listdlg('PromptString','Select ordinate image:','ListString', str);
if ~isOK, return; end

[imageX, image_infoX] = Data_Get(handles.figure1, selX);
[imageY, image_infoY] = Data_Get(handles.figure1, selY);
 mask = cMask.SafeGet(1, image_infoX.image_dim);

 dataX = imageX(mask(:));
 dataY = imageY(mask(:));
 
 figure;
 plot(dataX,dataY,'.')
 axis tight;
 xlabel(image_infoX.image_type,'interpreter','none');
 ylabel(image_infoY.image_type,'interpreter','none');
 
% --------------------------------------------------------------------
function nWindowMakeMovie_Callback(hObject, ~, handles)

dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'SourcePath', 'C:/');

[FileName,PathName] = uiputfile({ '*.avi', 'Movie files (*.avi)'; ...
  '*.*', 'All files (*.*)'},'Save file', old_path);

if ~(isequal(FileName,0) || isequal(PathName,0))
  
  outputVideo = VideoWriter(fullfile(PathName,FileName));
  outputVideo.FrameRate = 5;
  outputVideo.Quality = 100;
  open(outputVideo)
  
  for ii=1:handles.ImageDim(4)
    
    handles.projection(4) = ii;
    AfterProjectionChanged(handles); drawnow
    
    switch hObject
      case handles.nWindowMakeMovieAxis1, F = getframe(handles.axes_1);
      case handles.nWindowMakeMovieAxis2, F = getframe(handles.axes_2);
      case handles.nWindowMakeMovieAxis3, F = getframe(handles.axes_3);
      case handles.nWindowMakeMovieAxis4, F = getframe(handles.figure1);
    end
    writeVideo(outputVideo,F);
    
  end
  close(outputVideo)
end

% --------------------------------------------------------------------
function mContoursMath_Callback(hObject, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.options);
if cMask.are()
  MaskList = cMask.MaskNames();
  
  definputs(1).name = 'Operand 1';
  definputs(1).type = 3; % choice by N
  definputs(1).choices = MaskList;
  definputs(2).name = 'Operand 2';
  definputs(2).type = 3; % choice by N
  definputs(2).choices = MaskList;
  definputs(3).name = 'Operator';
  definputs(3).type = 2; % choice
  definputs(3).choices = {'AND', 'OR', 'NOTAND'};
  definputs(4).name = 'Result';
  definputs(4).type = 3; % choice by N
  definputs(4).choices = MaskList;
  
  a  = choiceinputdlg('Logical Operations',definputs);
  if ~isempty(a)
    in1 = a{1}; in2 = a{2}; selection = a{3}; out = a{4};
    cMask.Math2(in1, in2, out, selection, []);
  end
  AfterProjectionChanged(handles);
end

% --------------------------------------------------------------------
function mCenterDistance_Callback(hObject, ~, handles)
image_number = get(handles.pmImageType, 'Value');

image_size_opt = safeget(handles.options, 'size', []);
FOV = iff(safeget(image_size_opt, 'isFOV', 0), safeget(image_size_opt, 'FOV', handles.ImageSize(1)), handles.ImageSize(1));
if length(FOV) == 1, FOV = FOV*[1,1,1]; end

sl = Data_GetSlice(handles.figure1, image_number, 1, handles.projection);
find_distance_cm(1,sl, FOV)
sl = Data_GetSlice(handles.figure1, image_number, 2, handles.projection);
find_distance_cm(2,sl, FOV)
sl = Data_GetSlice(handles.figure1, image_number, 3, handles.projection);
find_distance_cm(3,sl, FOV)

% --------------------------------------------------------------------
function find_distance_cm(ax, slice, FOV)
threshold = max(slice(:)) * 0.35;
BW = slice > threshold;
CC = bwconncomp(BW);

dx = FOV(1:ndims(BW)) ./ size(BW);
dx2 = dx .* dx;
IM = [];
island = zeros(size(BW));
for ii=1:length(CC.PixelIdxList)
  island_idx = CC.PixelIdxList{ii};
  island(island_idx)=ii;
  [r, cMask] = find(island == ii);
  IM(ii).rc = [mean(r), mean(cMask)];
end

figure; imagesc(island); hold on
legend_text = {};
fprintf('\nDistances for axis %i ------\n', ax);
for ii=1:length(IM)
  for jj=ii+1:length(IM)
    dst = sqrt(dx2(1)*(IM(ii).rc(1) - IM(jj).rc(1))^2 + dx2(2)*(IM(ii).rc(2) - IM(jj).rc(2))^2);
    plot([IM(ii).rc(2), IM(jj).rc(2)], [IM(ii).rc(1), IM(jj).rc(1)],'linewidth',2)
    legend_text{end+1}=sprintf('%4.2f',dst);
    fprintf('  %i -> %i distance %f\n', ii, jj, dst);
  end
end
legend(legend_text)
axis image;
set(gca, 'YDir','normal')

function get_file_info(data)
if isfield(data, 'rec_info')
  if isfield(data.rec_info, 'fft')
    if isfield(data.rec_info.fft, 'profile_file')
      fprintf("profile_file = %s\n", data.rec_info.fft.profile_file);
    end
  end
end

% --------------------------------------------------------------------
function mRefitData_Callback(hObject, ~, handles)
% Locate raw data file

loaded_file = handles.ini.Directories.SourcePath;
save_pfile = loaded_file;

if exist(loaded_file, 'file') ~= 2
end

s1 = load(loaded_file);

if strfind(s1.file_type, 'FitImage')
    source_file = s1.source_image;
elseif strfind(s1.file_type, 'Image')
    fname = epri_filename(loaded_file, '', '');
    s1 = load(fname.p_file); 
    source_file = s1.source_image;
    [fpath, ffile, ffext] = fileparts(loaded_file);
    save_pfile = fullfile(fpath, ['p',ffile,ffext]);
end

if ~exist(source_file, 'file')
    [filename, pathname] = uigetfile( ...
       {'*.mat', 'All MATLAB Files (*.mat)'; ...
        '*.*',                   'All Files (*.*)'}, ...
        'Pick a RAW file',source_file);
    source_file = fullfile(pathname, filename);
end
s2 = load(source_file);
raw_data = s2.mat_recFXD;

switch s1.fit_data.Algorithm
    case 'T1_InvRecovery_3ParR1'
        % -------------------------------------------------------------------------
        % -------------- D A T A   F I T I N G  -----------------------------------
        % -------------------------------------------------------------------------
        if ndims(raw_data) ~=4, return; end
        res = inputdlg({'fit_mask_threshold', 'mask_proc_stage1', 'mask_proc_stage2', 'mask_proc_stage3', 'mask_proc_stage4'}, ...
            'Input parameters', 1, {num2str(s1.rec_info.fit.fit_mask_threshold), s1.rec_info.fit.mask_proc_stage1, s1.rec_info.fit.mask_proc_stage2, s1.rec_info.fit.mask_proc_stage3, s1.rec_info.fit.mask_proc_stage4});
        if ~isempty(res)
            s1.rec_info.fit.fit_mask_threshold = str2double(res{1});
            s1.rec_info.fit.mask_proc_stage1 = res{2};
            s1.rec_info.fit.mask_proc_stage2 = res{3};
            s1.rec_info.fit.mask_proc_stage3 = res{4};
            s1.rec_info.fit.mask_proc_stage4 = res{5};
            s1.rec_info.fit.fit_max_amp = 100;
            s1.source_image = source_file;
            [s1.fit_data] = epri_recovery_fit(raw_data, s1.raw_info.T1(:)'*1E6, s1.rec_info.fit);
            save(save_pfile,'-struct','s1');
            fprintf('File %s is saved.\n', loaded_file);
            
            Data_Load(loaded_file, handles, true, true);
            handles = guidata(handles.figure1);
            Mask_Check(handles.figure1, handles.ImageDim);
            UpdateMaskList(handles);
            AfterProjectionChanged(handles);            
        end
end

% --------------------------------------------------------------------
function mPermute_Callback(~, ~, handles)

image_number = get(handles.pmImageType, 'Value');

Images = getappdata(handles.figure1, 'Images');
Images{image_number}.Image = permute(Images{image_number}.Image, [1,2,4,3]);
setappdata(handles.figure1, 'Images', Images);


Masks = getappdata(handles.figure1, 'Masks');
Masks{image_number}.Mask = ones(size(Images{image_number}.Image));
Masks{image_number}.Mask = squeeze(Masks{image_number}.Mask(:,:,:,1));
setappdata(handles.figure1, 'Masks', Masks);

handles.ImageDim = size(Images{image_number}.Image);
handles.projection = floor(handles.ImageDim/2); 
guidata(handles.figure1, handles);

AfterProjectionChanged(handles);


% --------------------------------------------------------------------
function nWindowMakeSliceMovie_Callback(hObject, eventdata, handles)

if hObject == handles.nWindowMakeSliceMovie, return; end

dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'SourcePath', 'C:/');

[FileName,PathName] = uiputfile({ '*.avi', 'Movie files (*.avi)'; ...
  '*.*', 'All files (*.*)'},'Save file', old_path);

if ~(isequal(FileName,0) || isequal(PathName,0))
  
  outputVideo = VideoWriter(fullfile(PathName,FileName));
  outputVideo.FrameRate = 5;
  outputVideo.Quality = 100;
  open(outputVideo)
  
  image_number = get(handles.pmImageType, 'Value');
  for ii=1:handles.ImageDim(3)
    
%     handles.projection(4) = ii;
%     AfterProjectionChanged(handles); drawnow
    
    switch hObject
        case handles.nWindowMakeSliceMovieAxis1
           handles.projection(3) = ii;
            [slice, nslice] = Data_GetSlice(handles.figure1, image_number, 1, handles.projection);  
            AfterProjectionChanged(handles); drawnow
            getData
            F = getframe(handles.axes_1);
        case handles.nWindowMakeSliceMovieAxis2
            handles.projection(2) = ii;
            [slice, nslice] = Data_GetSlice(handles.figure1, image_number, 2, handles.projection);  
            AfterProjectionChanged(handles); drawnow
            F = getframe(handles.axes_2);
        case handles.nWindowMakeSliceMovieAxis3
            handles.projection(1) = ii;
            [slice, nslice] = Data_GetSlice(handles.figure1, image_number, 3, handles.projection);  
            AfterProjectionChanged(handles); drawnow
            F = getframe(handles.axes_3);
        case handles.nWindowMakeSliceMovieAxis4
%             handles.projection(4) = ii;
%             AfterProjectionChanged(handles); drawnow
%             F = getframe(handles.figure1);
    end
    if any(slice(:))
      writeVideo(outputVideo,F);
    end
    
  end
  close(outputVideo)
end

% --------------------------------------------------------------------
function nMenuCustom_Callback(hObject, eventdata, handles)
image_number = get(handles.pmImageType, 'Value');
Data = Data_Get3D(handles.figure1, image_number, handles.projection(4));

switch hObject
  case handles.mMenuCustom1, func = 'ibGUICustom3DProc_1';
  case handles.mMenuCustom2, func = 'ibGUICustom3DProc_2';
  case handles.mMenuCustom3, func = 'ibGUICustom3DProc_3';
  case handles.mMenuCustom4, func = 'ibGUICustom3DProc_4';
  case handles.mMenuCustom5, func = 'ibGUICustom3DProc_5';
end
if exist(func) == 2
  fprintf('Evaluation function %s\n', func);
  Data = feval(func, Data);
end
