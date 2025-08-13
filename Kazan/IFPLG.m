function varargout = IFPLG(varargin)
% IFPLG MATLAB code for IFPLG.fig
%      IFPLG, by itself, creates a new IFPLG or raises the existing
%      singleton*.
%
%      H = IFPLG returns the handle to a new IFPLG or the handle to
%      the existing singleton*.
%
%      IFPLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IFPLG.M with the given input arguments.
%
%      IFPLG('Property','Value',...) creates a new IFPLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before IFPLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to IFPLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help IFPLG

% Last Modified by GUIDE v2.5 11-Mar-2016 09:02:51

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @IFPLG_OpeningFcn, ...
                   'gui_OutputFcn',  @IFPLG_OutputFcn, ...
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


% --- Executes just before IFPLG is made visible.
function IFPLG_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to IFPLG (see VARARGIN)

% Choose default command line output for IFPLG
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes IFPLG wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = IFPLG_OutputFcn(hObject, eventdata, handles) 
% Get default command line output from handles structure
varargout{1} = handles.output;


% --------------------------------------------------------------------
function pushbutton1_Callback(hObject, eventdata, handles)

hand = guidata(handles.hh);

ax = hand.src.ax;
y  = hand.src.y;
npts = size(y,1);
npad = 8;
y(npts*npad,:) = 0;
dt = mean(diff(ax.x));
ax.x = (1:npad*npts)'*dt;

IF = eval(get(handles.eIF, 'string'))*1E6; % [Hz]
cut_off = eval(get(handles.eCutOff, 'string')); % [MHz]
FWHM = 1 / cut_off*1E-6; % [Hz]
ph = eval(get(handles.ePhase, 'string'))*pi/180; % [phase rad]


window_start = eval(get(handles.eFirstPoint, 'string'))*1E-9;
Fpnt = fix(window_start/dt);
Fpnt = max([Fpnt, 1]);

y = y(Fpnt:end,:);
t=ax.x;
t=t(Fpnt:end);
ref=exp(-1i*2*pi*IF*t+1i*ph);

filter=my_gaussian(t-mean(t),FWHM,0);

nTrace = size(y, 2); 
for ii=1:size(y, 2)
    yIFF(:,ii)=conv(y(:,ii).*ref,filter,'same');
    [v, yIF(:,ii)]=fftM(t,yIFF(:,ii));
end

h=v/2.8e6;
sp=zeros(size(yIF, 1), 1);
sp1=zeros(size(yIFF, 1), 1);
kk = [1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1];

% for ii=1:nTrace
%     sp = sp + kk(ii)*yIF(:,ii);
%     sp1 = sp1+ kk(ii)*yIFF(:,ii);
% end
% sp = sp / nTrace;

hand.out = {};

switch get(handles.pmDomain, 'value')
    case 3
        idx = v > -1e6*cut_off & v < cut_off*1e6;
        hand.out{1}.ax.x = v(idx)*1e-6 + IF*1e-6;
        hand.out{1}.ax.y = 1 : nTrace;
        hand.out{1}.y  = yIF(idx,:)*exp(-1i*ph);
    case 2
        idx = h > -10 & h < 10;
        hand.out{1}.ax.x = h(idx);
        hand.out{1}.ax.y = 1 : nTrace;
        hand.out{1}.y  = yIF(idx,:)*exp(-1i*ph);
    case 1
        hand.out{1}.ax.x = t(1:npts);
        hand.out{1}.ax.y= 1:nTrace;
        hand.out{1}.y  = yIFF(1:npts,:);
end

guidata(handles.hh, hand);
[path,name] = fileparts(get(handles.hh, 'FileName'));
eval([name '(''SetDataSource'', 2, hand)']);
