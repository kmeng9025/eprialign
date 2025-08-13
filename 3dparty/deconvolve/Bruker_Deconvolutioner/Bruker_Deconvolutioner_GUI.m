function varargout = Bruker_Deconvolutioner_GUI(varargin)
% BRUKER_DECONVOLUTIONER_GUI MATLAB code for Bruker_Deconvolutioner_GUI.fig
%      BRUKER_DECONVOLUTIONER_GUI, by itself, creates a new BRUKER_DECONVOLUTIONER_GUI or raises the existing
%      singleton*.
%
%      H = BRUKER_DECONVOLUTIONER_GUI returns the handle to a new BRUKER_DECONVOLUTIONER_GUI or the handle to
%      the existing singleton*.
%
%      BRUKER_DECONVOLUTIONER_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BRUKER_DECONVOLUTIONER_GUI.M with the given input arguments.
%
%      BRUKER_DECONVOLUTIONER_GUI('Property','Value',...) creates a new BRUKER_DECONVOLUTIONER_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Bruker_Deconvolutioner_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Bruker_Deconvolutioner_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Bruker_Deconvolutioner_GUI

% Last Modified by GUIDE v2.5 20-Jan-2016 17:11:58

% Begin initialization code - DO NOT EDIT

% addpath('Z:\CenterMATLAB\3dparty\deconvolve\Bruker_Deconvolutioner');

gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Bruker_Deconvolutioner_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @Bruker_Deconvolutioner_GUI_OutputFcn, ...
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


% --- Executes just before Bruker_Deconvolutioner_GUI is made visible.
function Bruker_Deconvolutioner_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Bruker_Deconvolutioner_GUI (see VARARGIN)
set(handles.fwhmedit, 'String', '50');
set(handles.raw_plot,'XTick',[]);
set(handles.ref_plot,'XTick',[]);
set(handles.deko_axis,'XTick',[]);

set(handles.radiobutton2, 'Value', 1);
set(handles.radiobutton3, 'Value', 0);

% Choose default command line output for Bruker_Deconvolutioner_GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Bruker_Deconvolutioner_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Bruker_Deconvolutioner_GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in loadraw.
function loadraw_Callback(hObject, eventdata, handles)
% hObject    handle to loadraw (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[FileName_raw,PathName_raw] = uigetfile({'*.dsc','Bruker file (*.dsc)';'*.exp','SpecMan file (*.exp)'});

if ~isequal(FileName_raw, 0)
  set(handles.path_raw, 'String', [PathName_raw FileName_raw]);
  [~,~,ext] = fileparts([PathName_raw,'\',FileName_raw]);
  
  switch upper(ext)
    case '.DSC'
      [img1,img2,img3]=eprload([PathName_raw,'\',FileName_raw]);
      handles.data.img1 = img1;
      handles.data.img2 = img2;
      handles.data.img3 = img3;
    case '.EXP'
      [ax, y]=kv_d01read([PathName_raw,'\',FileName_raw]);
      handles.data.img1 = ax.x;
      handles.data.img2 = y(:,2:end);
      handles.data.reference = y(:,1);
      %     handles.data.img3 = img3;
      
      % reference
      axes(handles.ref_plot);cla
      plot(handles.ref_plot,handles.data.reference);
      title('Ref EPR signal');
      xlim([0 size(handles.data.reference,1)]);
  end
  axes(handles.raw_plot);cla;
  plot(handles.raw_plot,handles.data.img2);
  title('Raw EPR signal');
  xlim([0 size(handles.data.img2,1)]);
  
  set(handles.slProjections, 'Max', size(handles.data.img2, 2), 'Value', 1)
  
  guidata(hObject, handles);
end

% --- Executes on button press in loadref.
function loadref_Callback(hObject, eventdata, handles)
% hObject    handle to loadref (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[FileName_ref,PathName_ref] = uigetfile('*.dsc','Select the Bruker ref file');
[~,ref]=eprload([PathName_ref,'\',FileName_ref]);
set(handles.path_ref, 'String', [PathName_ref FileName_ref]);
handles.data.reference = ref;

axes(handles.ref_plot);cla
plot(handles.ref_plot,ref);
title('Ref EPR signal');
xlim([0 size(ref,1)]);
guidata(hObject, handles);


function fwhmedit_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function fwhmedit_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on slider movement.
function slProjections_Callback(hObject, eventdata, handles)
deconvolution_Callback(hObject, eventdata, handles);


% --- Executes during object creation, after setting all properties.
function slProjections_CreateFcn(hObject, eventdata, handles)

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes on button press in deconvolution.
function deconvolution_Callback(hObject, eventdata, handles)

strfwhm=get(handles.fwhmedit, 'string');
fwhm = str2double(strfwhm); 

ref = handles.data.reference;
raw = handles.data.img2;
% raw=cumsum(raw, 1);
% ref=cumsum(ref, 1);
ntrace = size(raw,1); 
nslice = size(raw,2); 

if get(handles.radiobutton2, 'Value')
  % Gaussian
  pars.fwhm = fwhm;
  pars.function = 'gaussian';
else
  % Fermi
  pars.radius = fwhm/2;
  pars.roll   = str2double(get(handles.eRoll, 'string'));
  pars.function = 'Fermi';
end

offset = fix(str2double(get(handles.eRefOffset, 'string')));

P = epr_deconvolve(1:ntrace,raw,1:ntrace,circshift(ref,offset),pars);

bl = mean(raw);
raw_corrected = raw-repmat(bl, size(raw, 1), 1);
raw_integral = cumsum(raw_corrected);
% P = out_y;

fld = 1:ntrace; 

n = fix(get(handles.slProjections, 'value'));
plot(handles.deko_axis,fld, P(:,n)/max(P(:,n)), fld, raw_integral(:,n)/max(raw_integral(:,n)));
axes(handles.deko_axis);
xlim([0 size(P,1)]);
title(['Projections after deconvolution, FWHM = ',num2str(fwhm)]);
handles.results.projections = P;
guidata(hObject, handles);


% --- Executes on button press in radiobutton2.
function radiobutton2_Callback(hObject, eventdata, handles)
set(handles.radiobutton3, 'Value', 0)

% --- Executes on button press in radiobutton3.
function radiobutton3_Callback(hObject, eventdata, handles)
set(handles.radiobutton2, 'Value', 0)
