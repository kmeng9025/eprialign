function varargout = readAcqpGUI(varargin)
% READACQPGUI M-file for readAcqpGUI.fig
%      READACQPGUI, by itself, creates a new READACQPGUI or raises the existing
%      singleton*.
%
%      H = READACQPGUI returns the handle to a new READACQPGUI or the handle to
%      the existing singleton*.
%
%      READACQPGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in READACQPGUI.M with the given input arguments.
%
%      READACQPGUI('Property','Value',...) creates a new READACQPGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before readAcqpGUI_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to readAcqpGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help readAcqpGUI

% Last Modified by GUIDE v2.5 12-Aug-2005 13:55:46

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @readAcqpGUI_OpeningFcn, ...
    'gui_OutputFcn',  @readAcqpGUI_OutputFcn, ...
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


% --- Executes just before readAcqpGUI is made visible.
function readAcqpGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to readAcqpGUI (see VARARGIN)

% Choose default command line output for readAcqpGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

global info folder
folder=[];
info=[];
info.selected=1;
info.n=0;
info.folders={[]};
info.fields=strvcat('PulseProg: ','acqProtocol: ','txAtten: ','ReceiverGain: ',...
    'RepTime: ','EchoTime: ','nEchoes: ','RecovTime: ',...
    'nSlices: ','FOV: ','Matrix: ','SliceThick: ','SliceSep: ','SliceList: ',...
    'SliceOffset: ','ReadOffset: ','PhaseOffset: ','nAverages: ','ImageType: ','SlicepackVec: ',...
    'nEvolutionCycles: ','EvolutionDelay: ','FlipAngle: ','IdealFAat: ','BasicFreq: ',...
    'SpecWidth: ','ExcitationPulse: ','PulseShape: ','ReadOutDir: ',...
    'RareFactor: ','FatSat: ','Gating: ','wordType: ',...
    'rawWordType: ','ByteOrder: ','SaveTime: ','Filename: ',...
    'MusWt: ','subjOrientHF: ','subjOrientSP: ','PVver: ','dwiBvals: ');
if nargin==4 %varargin exists
    info.dir=varargin{1};
else
    info.dir='';
end

% UIWAIT makes readAcqpGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = readAcqpGUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in filesBox.
function filesBox_Callback(hObject, eventdata, handles)
% hObject    handle to filesBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns filesBox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from filesBox
global info
if info.n>0
    info.selected=get(hObject,'Value');
    set(handles.infoBox,'String',info.output(:,info.selected));
end

% --- Executes on button press in addButton.
function addButton_Callback(hObject, eventdata, handles)
% hObject    handle to addButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global info folder
if not(isequal(info.dir,''))
    addWSButton_Callback(hObject, eventdata, handles);
    return;
end
if isempty(folder)
    CH_Ddrive= dir('D:\mri_data\2009\');
    wd=cd;
    if (numel(CH_Ddrive)~=0)
        if strcmp(wd,'C:\Matlab\R2008b\work') || strcmp(wd,'C:\MATLAB\R2009a\work')
            folder=uigetdir('D:\mri_data\2009\','Select Folder Containing ACQP File');
        else
            folder=uigetdir(wd,'Select Folder Containing ACQP File');
        end
    else
        folder=uigetdir('X:\data\','Select Folder Containing ACQP File');
    end
else
    folder=uigetdir(folder,'Select Folder Containing ACQP File');
end
if isequal(folder,0)
    return;
end
if sum(strcmp(folder,info.folders))~=0
    return;
end
info.folders{info.n+1}=char(folder);
set(handles.filesBox,'String',info.folders);
info.n=info.n+1;
if info.n==1
    info.acqp=readAcqp(folder);
else
    info.acqp(info.n)=readAcqp(folder);
end
ii=struct2cell(info.acqp(info.n));
ii2=num2str(ii{1});
for k=2:length(ii)
    %     if k==28, keyboard, end
    temp=num2str(ii{k});
    ii2=strvcat(ii2,temp);
end

output=strcat(info.fields,ii2);
set(handles.infoBox,'String',output);
info.output(:,info.n)=cellstr(output);
info.selected=info.n;
set(handles.filesBox,'Value',info.selected);


% --- Executes on button press in removeButton.
function removeButton_Callback(hObject, eventdata, handles)
% hObject    handle to removeButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global info folder
if info.n<1
    errordlg('No ACQP files to remove!','Error!');
    return;
end
if info.selected~=info.n
    for j=info.selected+1:info.n
        info.acqp(j-1)=info.acqp(j);
        info.folders{j-1}=info.folders{j};
        info.output(:,j-1)=info.output(:,j);
    end
else
    info.selected=info.selected-1;
end
info.folders{info.n}=[];
info.acqp(info.n)=[];
info.output(:,info.n)=[];
info.n=info.n-1;
if info.n<1
    myOutput=' ';
    info.selected=1;
else
    myOutput=info.output(:,info.selected);
end
set(handles.filesBox,'String',info.folders,'Value',info.selected);
set(handles.infoBox,'String',myOutput);


% --- Executes on button press in addWSButton.
function addWSButton_Callback(hObject, eventdata, handles)
% hObject    handle to addWSButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% * * * * need to check to see if image is reconstructed
% otherwise readAcqpGUI will stumble

global info folder
wd=cd;
if isequal(info.dir,'')
    CH_Ddrive= dir('D:\mri_data\2009\');
    if (numel(CH_Ddrive)~=0)
        if strcmp(wd,'C:\Matlab\R2008b\work') || strcmp(wd,'C:\MATLAB\R2009a\work')
            folder=uigetdir('D:\mri_data\2009\','Select Folder Containing ACQP File');
        else
            folder=uigetdir(wd,'Select Folder Containing ACQP File');
        end
    else
        folder=uigetdir('X:\data\','Select Parent Folder');
    end
else
    folder=info.dir;
end
if isequal(folder,0)
    return;
end
j=char(folder);
if (j(length(j))~='\' & j(length(j))~='/')
    if (strfind(j,'\') ~= 0), j(length(j)+1)='\'; end;
    if (strfind(j,'/') ~= 0), j(length(j)+1)='/'; end;
    folder=j;
end
dirs=struct2cell(dir(folder));
dirs=char(dirs{1,:});
% Folders must start with a digit; only use correct ones
good=find(dirs(:,1)>='0' & dirs(:,1)<='9');
goodsize=size(good);
if ~goodsize(1)
    errordlg('Illegal parent directory!','Error!');
    return;
end
dirs=cellstr(dirs(good,:));
dirs=strcat(folder,dirs);
%sort directories
for n=1:size(dirs,1)
    testdir=dirs{n,1};
    %findNum=regexp(testdir,'[\/]');
    findNum=regexp(testdir,'\');
    sortDir(n)=str2num(testdir(findNum(end)+1:end));
end
[sortDr, idx] = sort(sortDir);
sortDirs=dirs(idx(1,:),1);
for w=1:length(sortDirs)
    if sum(strcmp(sortDirs(w),info.folders))==0
        info.folders{info.n+1}=char(sortDirs(w));
        info.n=info.n+1;
        disp(['Processing ', cell2mat(sortDirs(w))])
        if info.n==1
            info.acqp=readAcqp(sortDirs(w));
        else
            info.acqp(info.n)=readAcqp(sortDirs(w));
        end
        ii=struct2cell(info.acqp(info.n));
        ii2=num2str(ii{1});
        for k=2:length(ii)
            temp=num2str(ii{k});
            ii2=strvcat(ii2,temp);
        end
        output=strcat(info.fields,ii2);
        info.output(:,info.n)=cellstr(output);
        %disp(char(sortDirs(w)))
    end
end
set(handles.infoBox,'String',output);
set(handles.filesBox,'String',info.folders);
info.selected=info.n;
set(handles.filesBox,'Value',info.selected);


% --- Executes on button press in removeAllButton.
function removeAllButton_Callback(hObject, eventdata, handles)
% hObject    handle to removeAllButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global info folder

while info.n > 1
    removeButton_Callback(hObject, eventdata, handles)
end
removeButton_Callback(hObject, eventdata, handles)


% --- Executes on button press in prevButton.
function prevButton_Callback(hObject, eventdata, handles)
% hObject    handle to prevButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global info folder
info.selected=info.selected-1;
if info.selected<1
    info.selected=info.n;
end
set(handles.infoBox,'String',info.output(:,info.selected));
set(handles.filesBox,'Value',info.selected);


% --- Executes on button press in nextButton.
function nextButton_Callback(hObject, eventdata, handles)
% hObject    handle to nextButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global info folder
info.selected=info.selected+1;
if info.selected>info.n
    info.selected=1;
end
set(handles.infoBox,'String',info.output(:,info.selected));
set(handles.filesBox,'Value',info.selected);


% --- Executes on button press in renameButton.
function renameButton_Callback(hObject, eventdata, handles)
% hObject    handle to renameButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global info folder

%slashes=regexp(folder,'[\/]');
slashes=regexp(folder,'\');
szfp=length(slashes);
MusNameGs=folder(slashes(szfp-2)+1:slashes(szfp-1)-1);
myunbar=regexp(MusNameGs,'_');
if ~isempty(myunbar) && numel(myunbar)==2, MusNameGs=MusNameGs([1:myunbar(1)-1,myunbar(2)+1:end]);end
mouse=[char(inputdlg('What is the mouse''s name?','Mouse Name?',1,{MusNameGs})) '_'];
wd=cd;

ndyns=0;
for j=1:info.n
    folder=char(info.folders{j});
    cd(folder);
    %slashes=find(folder=='\' | folder=='/');
    last=slashes(length(slashes)-1:end);
    % number=folder(last(1)+1:last(2)-1);
    number=folder(last(2)+1:end);
    switch info.acqp(j).imageType
        case 'Axial'
            ort='Ax';
        case 'Sagittal'
            ort='Sg';
        case 'Coronal'
            ort='Cr';
    end
    if (strfind(lower(info.acqp(j).acqProtocol),'pilot')~=0)
        %number='';
        ort='';
        type='trip';
    elseif (strfind(info.acqp(j).acqProtocol,'FASTMAP')~=0)
        number='';
        ort='';
        type='FastMap';
    elseif (strfind(info.acqp(j).acqProtocol,'MAPSHIM')~=0)
%         number='';
%         ort='';
        type='MapShim';
    elseif (strfind(info.acqp(j).acqProtocol,'PRESS-')~=0)   
        ort='';
        type='PressW';
    elseif (strfind(info.acqp(j).acqProtocol,'')~=0)        
        ort='';
        type='SinglePulse';
    elseif (strfind(info.acqp(j).acqProtocol,'RARE')~=0)
        type='rare';
        if (strfind(info.acqp(j).acqProtocol,'RARE_8_bas3D')~=0)
            type='rare3D';
        elseif (strfind(info.acqp(j).acqProtocol,'RAREVTR_T1')~=0)
            type='T1map';
        end
        
    elseif (strfind(info.acqp(j).acqProtocol,'Spin')~=0)
        type='se';
    elseif (strfind(info.acqp(j).acqProtocol,'Grad')~=0)
        if info.acqp(j).nEvolutionCycles == 4
            type='DynTest';
            ort='';
            number='';
        elseif info.acqp(j).nEvolutionCycles > 4
            ndyns=ndyns+1;
            type=['Dyn' num2str(ndyns)];
            ort='';
            number='';
        else
            type='ge';
        end  %end if test dynamic
    elseif (logical(~isempty(strfind(info.acqp(j).acqProtocol,...
            'Dynamic'))) | logical(~isempty(strfind(info.acqp(j).acqProtocol,'FLASH'))))
        if info.acqp(j).nEvolutionCycles == 4
            type='DynTest';
            ort='';
            number='';
        elseif info.acqp(j).nEvolutionCycles > 4
            ndyns=ndyns+1;
            type=['Dyn' num2str(ndyns)];
            ort='';
            number='';
        else
            type='Flash';
        end  %end if dynmic test
    elseif (strfind(info.acqp(j).acqProtocol,'EPSI')~=0)
        type='EPSI';
    elseif (strfind(info.acqp(j).acqProtocol,'MGE_shortTE')~=0)
        type='EPSImge';
        disp('Assume EPSI, rename if plain MGE')
    elseif (strfind(info.acqp(j).acqProtocol,'MSME-T2-map')~=0)
        type='T2map';
    else  %if protocol is define above, the user can input a new protocol
        prompt = {'Current protocol','Your prefix'};
        dlg_title = 'Enter Prefix for file';
        num_lines= 2;
        defAns={info.acqp(j).acqProtocol,'unk'};
        newType = char(inputdlg(prompt,dlg_title,num_lines,defAns));
        type=strtrim(newType(2,:));
        %         ort='';
        %         number='';
    end  % test which protocol
    
    newname=[mouse type number ort '.fid'];
    test4fid=dir('fid');
    
    if ~isempty(test4fid)
        movefile('fid',newname);
    else
        % number will be blank if tripilot
        disp(['fid file not exist in ' number]);
    end
    cd('pdata\1');
    new2dname=[mouse type number ort '.img'];
    test4file=dir('2dseq');
    %    if strcmp(test4file.name,'2dseq')
    if ~isempty(test4file)
        movefile('2dseq',new2dname);
    else
        % number will be blank if tripilot
        disp(['2dseq file not exist in ' number]);
    end
end  %end for
cd(wd);
disp('Done renaming files');

% --- Executes on button press in writeButton.
function writeButton_Callback(hObject, eventdata, handles)
% hObject    handle to writeButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global info folder
wd=cd;
if info.n<1
    errordlg('No ACQP files to write!','Error!');
    return;
end
whichAcqps=questdlg('Write which ACQPs?','Which ACQPs?','All','Selected','Cancel','All');
if isequal(whichAcqps,'Cancel')
    return;
end
% cd('..')
% parfolder=cd;
saveFolder=char(uigetdir(wd,'Select save directory'));
if isequal(saveFolder,0)
    return;
end
%foldpos=regexp(folder,'[\/]');
foldpos=regexp(folder,'\');
szfp=length(foldpos);
MusNameGs=folder(foldpos(szfp-2)+1:foldpos(szfp-1)-1);
filename=inputdlg('Please enter a filename:','Filename?',1,{[MusNameGs,'_Acqp']});

cd(saveFolder);
file=char(strcat(filename,'.txt'));
fid=fopen(file,'w');

% number of fields
nf=size(info.fields,1);
if isequal(whichAcqps,'Selected')
    fprintf(fid,'%s\n',char(info.folders{info.selected}));
    for jjj=1:nf  
        fprintf(fid,'%s\n',char(info.output(jjj,info.selected)));
    end
else
    for k=1:info.n
        fprintf(fid,'%s\n',char(info.folders{k}));
        for jjj=1:nf  
            fprintf(fid,'%s\n',char(info.output(jjj,k)));
        end
        fprintf(fid,'%s\n\n','');
    end
end
fclose(fid);

% make CSV version, April 7 2009 CH
csvfile=char(strcat(filename,'.csv'));
% titles={'expnum    ', 'imgtype', 'tx0','tx1','RG','tr' ,'te ' ,'echo', 'slice','fov   ', 'array   ', 'slicethick ', 'SliceSep','nex','geom','nRev','paramnote'};
if csvfile ~=0
    %         csvpos=strfind(csvfile,'.csv');
    %         if (numel(csvpos) == 0)
    %             csvfile=strcat(csvfile, '.csv');
    %         end
    mytitles=cell(1,nf);
    for n=1:nf
        mydelim=regexp(info.fields(n,:),':');
        mytitles(n)={info.fields(n,1:mydelim-1)};
        clear mydelim
    end
    clear n
%     mytitles=fieldnames(info.acqp)';
    % remove titles that don't match InfoPath
    % 16 and 17 are now read and phase offsets CH 06-09-10
%     mytitles([1,8,14,15,18,20:end])=[];
    myList={'acqProtocol';'txAtten';'ReceiverGain';'RepTime';'EchoTime';'nEchoes';'nSlices';...
        'FOV';'Matrix';'SliceThick';'SliceSep';'nAverages';'ImageType';'nEvolutionCycles'};
    keepFields=ismember(mytitles,myList);
    mytitles(~keepFields) = [];    
%     mytitles([1,8,14:17,20,22:end])=[];
    % add tx0 and tx1and notes
    mytitles(2)={'tx0'};
    mytitles(end+1)={'tx1'};
    mytitles(end+1)={'ParamNote'};
    % shuffle so that tx1 is third
    mytitles=mytitles([1,2,end-1,3:end]);
    mytitles(end-1)=[];
    nfNew=size(mytitles,2);
    
    infopathdata=struct2cell(info.acqp');
    % remove fields not in InfoPath and concatenate the rest for the note
    % field
    % 16 and 17 are now read and phase offsets CH 06-09-10
    % 
    infopathdata(~keepFields,:)=[];
%     infopathdata([1,8,14,15,20,22:end],:)=[];
    infopathdata=infopathdata([1,2,2:end],:);
   % 2 to 13 & 15 need num2str
    for x=1:info.n
        % split tx 0 and tx1
        infopathdata{3,x} = num2str(round((infopathdata{2,x}(2))*10)/10);
        infopathdata{2,x} = num2str(round((infopathdata{2,x}(1))*10)/10);
        infopathdata{15,x}=num2str(infopathdata{15,x});
        infopathdata{nfNew,x}=char(strtrim(info.output(~keepFields,x)'))';
%         infopathdata{nfNew,x}=char(deblank(info.output([1,15:17,22:23,25:31],x)'))';
        for xx=4:13, infopathdata{xx,x}=num2str(infopathdata{xx,x}); end
    end
    infopathdata=infopathdata';
    clear x xx
    
    testfn = exist(csvfile,'file');  %check if file exist, then append
    if testfn == 0
        csvwrite_cp(csvfile, mytitles);
    else
        csvappend(csvfile, mytitles);
    end
    csvappend(csvfile, infopathdata);
end

cd(wd);
disp('Done making Info file');

% --- Executes on button press in quitButton.
function quitButton_Callback(hObject, eventdata, handles)
% hObject    handle to quitButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close(gcf);
vars = evalin('base','who');
if ismember('saddleCoil',vars)
    evalin('base','clear saddleCoil');
end
clear vars
clear global info folder


%stuff that doesn't need to be edited
% --- Executes during object creation, after setting all properties.
function filesBox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to filesBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on selection change in infoBox.
function infoBox_Callback(hObject, eventdata, handles)
% hObject    handle to infoBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns infoBox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from infoBox

% --- Executes during object creation, after setting all properties.
function infoBox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to infoBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
