function varargout = ImageCompareOptionsDLG(varargin)
% IMAGECOMPAREOPTIONSDLG M-file for ImageCompareOptionsDLG.fig
%      IMAGECOMPAREOPTIONSDLG, by itself, creates a new IMAGECOMPAREOPTIONSDLG or raises the existing
%      singleton*.
%
%      H = IMAGECOMPAREOPTIONSDLG returns the handle to a new IMAGECOMPAREOPTIONSDLG or the handle to
%      the existing singleton*.
%
%      IMAGECOMPAREOPTIONSDLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IMAGECOMPAREOPTIONSDLG.M with the given input arguments.
%
%      IMAGECOMPAREOPTIONSDLG('Property','Value',...) creates a new IMAGECOMPAREOPTIONSDLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ImageCompareOptionsDLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ImageCompareOptionsDLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ImageCompareOptionsDLG

% Last Modified by GUIDE v2.5 27-Jan-2015 09:51:12

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ImageCompareOptionsDLG_OpeningFcn, ...
                   'gui_OutputFcn',  @ImageCompareOptionsDLG_OutputFcn, ...
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
function ImageCompareOptionsDLG_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ImageCompareOptionsDLG (see VARARGIN)

% Choose default command line output for ImageCompareOptionsDLG
handles.output  = hObject;
if ~isempty(varargin)
  handles.options = varargin{1};
else
  handles.options = [];
end

% size
opt_size = safeget(handles.options, 'size', []);
if ~iscell(opt_size), opt_size = {opt_size}; end
im_size = length(opt_size);
str = {};
for ii=1:im_size
  opt_size{ii}.isFOV = safeget(opt_size{ii}, 'isFOV', 1);
  opt_size{ii}.Dim  = safeget(opt_size{ii}, 'Dim', [64, 64, 64]);
  opt_size{ii}.Size = safeget(opt_size{ii}, 'Size', opt_size{ii}.Dim);
  opt_size{ii}.FOV = iff(opt_size{ii}.isFOV, safeget(opt_size{ii}, 'FOV', opt_size{ii}.Size), opt_size{ii}.Size);
  str{ii} = sprintf('Image %i', ii);
end
set(handles.pmSizeOfImages, 'String', str, 'Value', 1);
handles.options.size = opt_size;

% size
opt_roi = safeget(handles.options, 'roi', []);
opt_roi.isROI = safeget(opt_roi, 'isROI', 0);
opt_roi.ROI = safeget(opt_roi, 'ROI', opt_size{1}.FOV);
handles.options.roi = opt_roi;

set(handles.pmDataTypeSelector, 'String', handles.options.ActiveFields);
handles.previous_data_type = -1;
pmDataTypeSelector_Callback(handles.pmDataTypeSelector, [], handles)
handles = guidata(hObject);

% mask settings
opt_mask = safeget(handles.options, 'mask', []);
opt_mask.ROI = safeget(opt_mask, 'ROI', []);
opt_mask.erosion = set_to_limit(safeget(opt_mask, 'erosion', 0), -5, 0);
opt_mask.ROI.erosion = set_to_limit(safeget(opt_mask.ROI, 'erosion', 0), -2, 2);
opt_mask.source = set_to_limit(safeget(opt_mask, 'source', 0), 0, 2); % as is 
opt_mask.isCommon = safeget(opt_mask, 'isCommon', 0);
handles.options.mask = opt_mask;

% controls initialization
handles.CurrentImage = 1;
set(handles.cb_isFOV, 'Value', fix_bool(opt_size{handles.CurrentImage}.isFOV));
FOV = opt_size{handles.CurrentImage}.FOV;
if numel(FOV) < 2, FOV(2) = FOV(1); end
if numel(FOV) < 3, FOV(3) = FOV(2); end
set(handles.eFOVx, 'String', num2str(FOV(1)));
set(handles.eFOVy, 'String', num2str(FOV(2)));
set(handles.eFOVz, 'String', num2str(FOV(3)));

set(handles.cb_isROI, 'Value', fix_bool(opt_roi.isROI));
set(handles.eROI, 'String', num2str(max(opt_roi.ROI)));

set(handles.pmMaskErosion, 'Value', 1-opt_mask.erosion);
set(handles.pmROIerosion, 'Value', 3-opt_mask.ROI.erosion);
set(handles.cbMaskCommon, 'Value', fix_bool(opt_mask.isCommon));
set(handles.pmMaskSource, 'Value', opt_mask.source + 1);

set(handles.eDescription, 'String', safeget(handles.options,'description',''));

% Update handles structure
guidata(hObject, handles);
uiwait(handles.figure1)

% --------------------------------------------------------------------
function x = fix_bool(x)
if x > 0.5, x=1; else x=0; end

% --------------------------------------------------------------------
function x = set_to_limit(x, min_lim, max_lim) 
x = max(x,min_lim);
x = min(x,max_lim);

% --------------------------------------------------------------------
function varargout = ImageCompareOptionsDLG_OutputFcn(hObject, eventdata, handles) 
if isfield(handles, 'OptionsOk')
  handles.options.description = get(handles.eDescription, 'String');

  handles.options.size{handles.CurrentImage}.isFOV = get(handles.cb_isFOV, 'Value');
  try 
    FOV(1)=str2double(get(handles.eFOVx, 'String')); 
    FOV(2)=str2double(get(handles.eFOVy, 'String')); 
    FOV(3)=str2double(get(handles.eFOVz, 'String')); 
    handles.options.size{handles.CurrentImage}.FOV = FOV;
  catch err, 
    handles.options.size{handles.CurrentImage}.FOV = [1,1,1];
  end

  handles.options.roi.isROI = get(handles.cb_isROI, 'Value');
  try handles.options.roi.ROI(1:3)=str2double(get(handles.eROI, 'String')); catch err, handles.options.roi.isROI = 0; end
  handles.options.roi.ROI = min([handles.options.roi.ROI;handles.options.size{handles.CurrentImage}.FOV], [], 1);
  
  handles.options.mask.erosion = 1-get(handles.pmMaskErosion, 'Value');
  handles.options.mask.ROI.erosion = 3-get(handles.pmROIerosion, 'Value');
  handles.options.mask.isCommon = get(handles.cbMaskCommon, 'Value');
  handles.options.mask.source = get(handles.pmMaskSource, 'Value') - 1;

  pmDataTypeSelector_Callback(handles.pmDataTypeSelector, [], handles)
  handles = guidata(hObject);

  varargout{1} = handles.options;
  varargout{2} = handles.OptionsOk;
  delete(hObject);
else
  varargout{1} = {};
  varargout{2} = 0;
end

% --------------------------------------------------------------------
function pbOk_Callback(hObject, eventdata, handles)
handles.OptionsOk = 1;
guidata(hObject, handles);
uiresume

% --------------------------------------------------------------------
function pbCancel_Callback(hObject, eventdata, handles)
handles.OptionsOk = 0;
guidata(hObject, handles);
uiresume

% --------------------------------------------------------------------
function pmDataTypeSelector_Callback(hObject, eventdata, handles)
data_type = get(hObject, 'Value');

% load previous settings from the panel
if handles.previous_data_type >= 1
  new_opt = safeget(handles.options, handles.options.ActiveFields{handles.previous_data_type}, []);
  new_opt.isLowlim = get(handles.cbAmpLowlim, 'Value');
  new_opt.isHighlim = get(handles.cbAmpHighlim, 'Value');
  try new_opt.lim(1)=str2double(get(handles.eAmpLowLim, 'String')); catch err, end
  try new_opt.lim(2)=str2double(get(handles.eAmpHighLim, 'String')); catch err, end
  new_opt.scale_minmax(1) = iff(new_opt.isLowlim, new_opt.lim(1), new_opt.minmax(1));
  new_opt.scale_minmax(2) = iff(new_opt.isHighlim, new_opt.lim(2), new_opt.minmax(2));
  new_opt.isShowLowlim = get(handles.cbShowAmpLowlim, 'Value');
  new_opt.isShowHighlim = get(handles.cbShowAmpHighlim, 'Value');
  new_opt.isStatLowlim = get(handles.cbStatAmpLowlim, 'Value');
  new_opt.isStatHighlim = get(handles.cbStatAmpHighlim, 'Value');
  new_opt.show_minmax = new_opt.minmax;
  try new_opt.show_lim(1)=str2double(get(handles.eShowAmpLowLim, 'String')); catch err, end
  try new_opt.show_lim(2)=str2double(get(handles.eShowAmpHighLim, 'String')); catch err, end
  try new_opt.stat_lim(1)=str2double(get(handles.eStatAmpLowLim, 'String')); catch err, end
  try new_opt.stat_lim(2)=str2double(get(handles.eStatAmpHighLim, 'String')); catch err, end
  new_opt.show_minmax(1) = iff(new_opt.isShowLowlim, new_opt.show_lim(1), new_opt.minmax(1));
  new_opt.show_minmax(2) = iff(new_opt.isShowHighlim, new_opt.show_lim(2), new_opt.minmax(2));
  new_opt.stat_minmax(1) = iff(new_opt.isStatLowlim, new_opt.stat_lim(1), new_opt.minmax(1));
  new_opt.stat_minmax(2) = iff(new_opt.isStatHighlim, new_opt.stat_lim(2), new_opt.minmax(2));
  handles.options.(handles.options.ActiveFields{handles.previous_data_type}) = new_opt;
  guidata(hObject, handles);
end
handles.previous_data_type = data_type;
guidata(hObject, handles);

% set new settings to the panel 
new_opt = safeget(handles.options, handles.options.ActiveFields{data_type}, []);
set(handles.cbAmpLowlim, 'Value', fix_bool(safeget(new_opt, 'isLowlim', 0)));
set(handles.cbAmpHighlim, 'Value', fix_bool(safeget(new_opt, 'isHighlim', 0)));
new_opt.lim = safeget(new_opt, 'lim', new_opt.minmax);
new_opt.lim(1) = iff(safeget(new_opt,'isLowlim',0), new_opt.lim(1), new_opt.minmax(1));
new_opt.lim(2) = iff(safeget(new_opt,'isHighlim',0), new_opt.lim(2), new_opt.minmax(2));
new_opt.show_lim = safeget(new_opt, 'show_lim', new_opt.minmax);
new_opt.stat_lim = safeget(new_opt, 'stat_lim', new_opt.minmax);
set(handles.eAmpLowLim, 'String', num2str(new_opt.lim(1)));
set(handles.eAmpHighLim, 'String', num2str(new_opt.lim(2)));
set(handles.cbShowAmpLowlim, 'Value', fix_bool(safeget(new_opt, 'isShowLowlim', 0)));
set(handles.cbShowAmpHighlim, 'Value', fix_bool(safeget(new_opt, 'isShowHighlim', 0)));
set(handles.cbStatAmpLowlim, 'Value', fix_bool(safeget(new_opt, 'isStatLowlim', 0)));
set(handles.cbStatAmpHighlim, 'Value', fix_bool(safeget(new_opt, 'isStatHighlim', 0)));
set(handles.eShowAmpLowLim, 'String', num2str(new_opt.show_lim(1)));
set(handles.eShowAmpHighLim, 'String', num2str(new_opt.show_lim(2)));
set(handles.eStatAmpLowLim, 'String', num2str(new_opt.stat_lim(1)));
set(handles.eStatAmpHighLim, 'String', num2str(new_opt.stat_lim(2)));

% --------------------------------------------------------------------
function pmSizeOfImages_Callback(hObject, eventdata, handles)
handles.options.size{handles.CurrentImage}.isFOV = get(handles.cb_isFOV, 'Value');
try handles.options.size{handles.CurrentImage}.FOV=str2double(get(handles.eFOVx, 'String')); catch err, end

handles.CurrentImage = get(handles.pmSizeOfImages, 'Value');
set(handles.cb_isFOV, 'Value', fix_bool(handles.options.size{handles.CurrentImage}.isFOV));
set(handles.eFOVx, 'String', num2str(handles.options.size{handles.CurrentImage}.FOV));
guidata(hObject, handles);

% --------------------------------------------------------------------
function pbResetFOV_Callback(hObject, eventdata, handles)
opt_size = safeget(handles.options, 'size', []);
FOV = opt_size{handles.CurrentImage}.Size;
if numel(FOV) < 2, FOV(2) = FOV(1); end
if numel(FOV) < 3, FOV(3) = FOV(2); end
set(handles.eFOVx, 'String', num2str(FOV(1)));
set(handles.eFOVy, 'String', num2str(FOV(2)));
set(handles.eFOVz, 'String', num2str(FOV(3)));

% --------------------------------------------------------------------
function pbSetVoxels_Callback(hObject, eventdata, handles)
opt_size = safeget(handles.options, 'size', []);
FOV = opt_size{handles.CurrentImage}.Dim;
if numel(FOV) < 2, FOV(2) = FOV(1); end
if numel(FOV) < 3, FOV(3) = FOV(2); end
set(handles.eFOVx, 'String', num2str(FOV(1)));
set(handles.eFOVy, 'String', num2str(FOV(2)));
set(handles.eFOVz, 'String', num2str(FOV(3)));

