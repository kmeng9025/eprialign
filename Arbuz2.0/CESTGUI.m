function varargout = CESTGUI(varargin)
% CESTGUI MATLAB code for CESTGUI.fig
%      CESTGUI, by itself, creates a new CESTGUI or raises the existing
%      singleton*.
%
%      H = CESTGUI returns the handle to a new CESTGUI or the handle to
%      the existing singleton*.
%
%      CESTGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CESTGUI.M with the given input arguments.
%
%      CESTGUI('Property','Value',...) creates a new CESTGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CESTGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CESTGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CESTGUI

% Last Modified by GUIDE v2.5 06-Mar-2023 15:23:35

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CESTGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @CESTGUI_OutputFcn, ...
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


% --- Executes just before CESTGUI is made visible.
function CESTGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to CESTGUI (see VARARGIN)

% Choose default command line output for CESTGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes CESTGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = CESTGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------------
function edit2_Callback(hObject, eventdata, handles)


% --------------------------------------------------------------------------
function edit2_CreateFcn(hObject, eventdata, handles)

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------------
function pushbutton1_Callback(hObject, eventdata, handles)

idxRare = -1;
idxCEST = -1;
CESTscale = [1,1,1];

CESTSignature = handles.eCESTSignature.String;

for ii=1:12
  folder = GetFolderName(handles.edit2.String, ii);
  if exist(folder, 'file') == 7
    fname = fullfile(folder, 'pdata\1\2dseq.');
    if exist(fname, 'file') == 2
       [~, pars] = epr_LoadBrukerMRI(fname);
       fprintf('FOLDER: %s\n', folder);
       fprintf('ACQ_method: %s\n', pars.acqp.ACQ_method);
       fprintf('RECO_inp_size: %i %i\n', pars.reco.RECO_inp_size);
       if idxCEST == -1 && contains(pars.acqp.ACQ_method, CESTSignature) && pars.reco.RECO_size(1) == 64
         idxCEST = ii;
         CESToffset = pars.acqp.ACQ_slice_offset;
         FOV = pars.reco.RECO_fov*10;
         sz = pars.reco.RECO_size;
         CESTscale = FOV./sz;
         CESTscale(3) = mean(diff(CESToffset));
       elseif idxRare == -1 && contains(pars.acqp.ACQ_method, 'RARE') && pars.reco.RECO_size(1) == 256
         idxRare = ii;
         RAREoffset = pars.acqp.ACQ_slice_offset;
       end
    end
  end
end
fprintf('idxRare = %i\n', idxRare);
if exist('RAREoffset', 'var'), fprintf('RAREoffset = %f\n', RAREoffset(1)); end
fprintf('idxCEST = %i\n', idxCEST);
if exist('CESToffset', 'var'), fprintf('CESToffset = %f\n', CESToffset(1)); end

% save offsets into file
if ~isempty(handles.edit1.String)
  save(handles.edit1.String, 'RAREoffset', 'CESToffset', 'CESTscale', '-append')
end

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
[filename, pathname, filterindex] = uigetfile('*.mat', 'Pick a MATLAB file');
handles.edit1.String = fullfile(pathname, filename);

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
[pathname] = uigetdir('*.mat', 'Pick a MATLAB code file');
handles.edit2.String = pathname;


% --------------------------------------------------------------------------
function pushbutton4_Callback(hObject, eventdata, handles)

fprintf('\n\n------------------------------------------------------------------\n');
for ii=1:12
  folder = GetFolderName(handles.edit2.String, ii);
  if exist(folder, 'file') == 7
    fname = fullfile(folder, 'pdata\1\2dseq.');
    fname2 = fullfile(folder, 'pdata\1\2dseq.img');

    if exist(fname, 'file') == 2
      [~, pars] = epr_LoadBrukerMRI(fname);
      fprintf('FOLDER: %c\n', folder(end));
      fprintf('  ACQ_method: %s\n', pars.acqp.ACQ_method);
      fprintf('  RECO_inp_size: %i %i\n', pars.reco.RECO_inp_size);
      fmt=['  ACQ_slice_offset: ' repmat(' %4.1f',1,numel(pars.acqp.ACQ_slice_offset))];
      fprintf([fmt,'\n'], pars.acqp.ACQ_slice_offset);
    end
    if exist(fname2, 'file') == 2
      [~, pars] = epr_LoadBrukerMRI(fname2);
      fprintf('FOLDER: %c\n', folder(end));
      fprintf('  ACQ_method: %s\n', pars.acqp.ACQ_method);
      fprintf('  RECO_inp_size: %i %i\n', pars.reco.RECO_inp_size);
      fmt=['  ACQ_slice_offset: ' repmat(' %4.1f',1,numel(pars.acqp.ACQ_slice_offset))];
      fprintf([fmt,'\n'], pars.acqp.ACQ_slice_offset);
    end
  end
end
if exist('RAREoffset', 'var'), fprintf('  RAREoffset = %f\n', RAREoffset(1)); end
if exist('CESToffset', 'var'), fprintf('  CESToffset = %f\n', CESToffset(1)); end
% --------------------------------------------------------------------------

function newpath = GetFolderName(fpath, index)

newpath = fullfile(fpath, num2str(index));

D = dir([newpath,'_*.*']);

if length(D) == 1
    newpath = fullfile(D.folder, D.name);
end




