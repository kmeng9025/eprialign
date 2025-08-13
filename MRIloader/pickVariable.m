function varargout = pickVariable(varargin)
% variable_out=pickVariable(varloc,numselect,myfile);
% varloc=1 means variable location is the workspace, varloc =0 means from file
% numselect= number of expected variables out
% myfile = file to pick variables from if varloc=0
%
% PICKVARIABLE M-file for pickVariable.fig
%      PICKVARIABLE, by itself, creates a new PICKVARIABLE or raises the existing
%      singleton*.
%
%      H = PICKVARIABLE returns the handle to a new PICKVARIABLE or the handle to
%      the existing singleton*.
%
%      PICKVARIABLE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PICKVARIABLE.M with the given input arguments.
%
%      PICKVARIABLE('Property','Value',...) creates a new PICKVARIABLE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before pickVariable_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to pickVariable_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help pickVariable

% Last Modified by GUIDE v2.5 16-Feb-2007 12:17:15

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @pickVariable_OpeningFcn, ...
    'gui_OutputFcn',  @pickVariable_OutputFcn, ...
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


% --- Executes just before pickVariable is made visible.
function pickVariable_OpeningFcn(hObject, eventdata, handles,varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to pickVariable (see VARARGIN)

% Choose default command line output for pickVariable
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);



% example from help\techdoc\creating_guis\examples\lb.m
% Populate the listbox
%update_listbox(handles)
%set(handles.listbox1,'Value',[])
global pickedVar varloc myfile numselect userbrowsed

%pickedVar ='initial_junk';
pickedVar={};
varloc = varargin{1};  %flag for variable from base (varloc=1) or from file (varloc=0)
numselect = varargin{2}; %number of allowed selections in listbox
myfile = varargin{3};  %file to be loaded if varloc ~= 1
if size(varargin,2)==4
    msgpassed=varargin{4};
    set(handles.msgPassed,'String',msgpassed)
else
    set(handles.msgPassed,'Visible','off')
end % display text for user
userbrowsed=0;  %initialize flag in case user changes from base to browse

if  varloc == 1;
    vars = evalin('base','who');
    set(handles.listbox1,'String',vars)
else
    vars = load(myfile);
    fn = fieldnames(vars);
    set(handles.listbox1,'String',fn)
end
if isempty(varargin{2})
    set(handles.textNmV,'Visible','off')
    set(handles.inputNumV,'Visible','off')
else
    set(handles.inputNumV,'String',num2str(numselect))
end
% UIWAIT makes pickVariable wait for user response (see UIRESUME)
uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = pickVariable_OutputFcn(hObject, eventdata, handles)
global pickedVar varloc myfile numselect userbrowsed
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
% varargout{1} = handles.output;

%  disp(['varargout ',pickedVar])
varargout{1} = pickedVar;
%if (userbrowsed == 1)
varargout{2} = userbrowsed;  % return the file used, in case the user browsed
% for a different file
%end
clear global pickedVar varloc myfile numselect userbrowsed
delete(handles.figure1)


% --- Executes on button press in browsebutton.
function browsebutton_Callback(hObject, eventdata, handles)
global pickedVar varloc myfile numselect userbrowsed

[mymat,dirstring]=uigetfile({'*.mat;*.MAT','Registration data (*.mat)';'*.*','All files (*.*)'},'Select new MAT file');
if (mymat ~= 0)
    userbrowsed = [dirstring, mymat];
    vars = load(userbrowsed);
    fn = fieldnames(vars);
    set(handles.listbox1,'String',fn)
    %userbrowsed=1;
end

% --- Executes on button press in QuitPV.
function QuitPV_Callback(hObject, eventdata, handles)
global pickedVar varloc myfile numselect userbrowsed
%disp(['quit ', pickedVar])
list_entries = get(handles.listbox1,'String');
index_selected = get(handles.listbox1,'Value');
if length(index_selected) ~= numselect
    errordlg(['You must select ' num2str(numselect) ' variable(s)'],'Pay Attention','modal');
    return;
else
    %list_entries
    %whos list_entries
    %whos index_selected
    pickedVar = {list_entries{index_selected}};
    % disp(['listbox ', pickedVar])
end

guidata(handles.figure1, handles);
uiresume(handles.figure1);
close(gcf);

% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)

guidata(handles.figure1, handles)
uiresume(handles.figure1)


% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
global pickedVar varloc myfile numselect userbrowsed

% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end