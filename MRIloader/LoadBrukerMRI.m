function varargout = LoadBrukerMRI(varargin)
% LOADBRUKERMRI M-file for LoadBrukerMRI.fig
%      LOADBRUKERMRI, by itself, creates a new LOADBRUKERMRI or raises the existing
%      singleton*.
%
%      H = LOADBRUKERMRI returns the handle to a new LOADBRUKERMRI or the handle to
%      the existing singleton*.
%
%      LOADBRUKERMRI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in LOADBRUKERMRI.M with the given input arguments.
%
%      LOADBRUKERMRI('Property','Value',...) creates a new LOADBRUKERMRI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before LoadBrukerMRI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to LoadBrukerMRI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help LoadBrukerMRI

% Last Modified by GUIDE v2.5 04-Feb-2010 14:37:49

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @LoadBrukerMRI_OpeningFcn, ...
                   'gui_OutputFcn',  @LoadBrukerMRI_OutputFcn, ...
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


% --- Executes just before LoadBrukerMRI is made visible.
function LoadBrukerMRI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to LoadBrukerMRI (see VARARGIN)

% % default positions
% % head, feet, prone, supine
% patOrient=[0 1 1 0];
% set(handles.head1st,'Value',patOrient(1))
% set(handles.ft1st,'Value',patOrient(2))
% set(handles.pronPos,'Value',patOrient(3))
% set(handles.supPos,'Value',patOrient(4))
% Choose default command line output for LoadBrukerMRI
set(handles.numSeriestxt,'Visible','Off')
set(handles.numSeries,'Visible','Off')
set(handles.musWt_txt,'Visible','Off')
set(handles.musWt,'Visible','Off')
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes LoadBrukerMRI wait for user response (see UIRESUME)
% uiwait(handles.LoadBrukerMRI_fig);


% --- Outputs from this function are returned to the command line.
function varargout = LoadBrukerMRI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% % ___make radio buttons mutual exclusive______
% function radioselect(hObject,rbTag)
% hr = findobj('tag', rbTag);
% set(hr, 'value', 0)
% set(hObject, 'value', 1)


% --- Executes on button press in myQuit.
function myQuit_Callback(hObject, eventdata, handles)
% hObject    handle to myQuit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close(handles.LoadBrukerMRI_fig);

% --- Executes on button press in myLoadImages.
function myLoadImages_Callback(hObject, eventdata, handles)
% hObject    handle to myLoadImages (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(handles.LoadBrukerMRI_fig,'Visible', 'Off')
set(handles.LoadBrukerMRI_fig,'HandleVisibility','Off')
drawnow;
% default {'FeetFirst','Prone'}
% fixed reading radiobutton so Feet1st correct
% CH 06-09-08
if get(handles.ft1st,'Value')==1
    myOptions.patOrient{1}={'FeetFirst'};
elseif get(handles.head1st,'Value')==1
    myOptions.patOrient{1}={'HeadFirst'};
end
if get(handles.pronPos,'Value')==1
    myOptions.patOrient{2}={'Prone'};
elseif get(handles.supPos,'Value')==1
    myOptions.patOrient{2}={'Supine'};
end
if get(handles.saddlecbx,'Value')==1
    saddleCoil='Yes';
else
    saddleCoil='No';
end
assignin('base','saddleCoil',saddleCoil);

if get(handles.hiGdcbx,'Value')==1
    fixDoseYN='Yes';
else
    fixDoseYN='No';
end
assignin('base','fixDoseYN',fixDoseYN);

if get(handles.BrukimgTypeA,'Value')==1
    % load anatomic image
    Imgs2Save=str2num(get(handles.Img2Save,'String'));
    MRIpng(Imgs2Save,'patOrient',myOptions.patOrient);
elseif get(handles.BrukimgTypeD,'Value')==1
    % load dynamic image
    myOptions.numSeries=str2num(get(handles.numSeries,'String'));
    myOptions.musWt=str2num(get(handles.musWt,'String'));
    makeDCE(myOptions);
elseif get(handles.BrukimgTypeT,'Value')==1
    % load T1 map
    wd=cd;
    if strcmp(wd,'C:\Matlab\R2008b\work') || strcmp(wd,'C:\MATLAB\R2009a\work')
        CH_Ddrive= dir('D:\MRI_data\2009\RalphClinicalTNF\');
        if (numel(CH_Ddrive)~=0)
            cd('D:\mri_data\2009\')
        else
            CH_Qdrive= dir('q:\mri_data\2009\');
            if (numel(CH_Qdrive)~=0)
                cd('Q:\MRI_data\2009\')
            end
        end
    end
    [t1files,t1path]=uigetfile({'*.img;*.IMG','Raw data (*.img)';'*.*',...
        'All files (*.*)'},'Choose T1 image :');
    if t1files==0 & t1path==0
        error('user canceled')
    end
    
    mySlash=regexp(t1path,'\');
    MusNameGs=t1path(mySlash(end-5)+1:mySlash(end-4)-1);
    myunbar=regexp(MusNameGs,'_');
    if ~isempty(myunbar) && numel(myunbar)==2, MusNameGs=MusNameGs([1:myunbar(1)-1,myunbar(2)+1:end]);end
    musID=[];
    while isempty(musID)
        musIDfn = inputdlg({'Mouse ID'},'Input Mouse ID',1,{MusNameGs});
        musIDfn = musIDfn{1};
        if isempty(musIDfn)
            warndlg('Mouse ID cannot be empty')
            musID=[];
            continue
        elseif isreal(str2double(musIDfn(1))) && ~(strcmpi(musIDfn(1),'j') || strcmpi(musIDfn(1),'i')) && ~isnan(str2double(musIDfn(1)))
            warndlg('Cannot have number as variable or start with a number')
            musID=[];
            continue
        elseif strfind(musIDfn,'-')
            warndlg('Mouse ID cannot have ''-''')
            musID=[];
            continue
        elseif strfind(musIDfn,' ')
            warndlg('Mouse ID cannot have ''a space''')
            musID=[];
            continue
        else
            musID=musIDfn;
        end
    end % while loop
    
    t1FN = [t1path,t1files];
    numFids = str2num(char(inputdlg({'No. Fids+1'},'How many fiducials?',1,{'2 2 2'})));
    t1Thresh = str2num(char(inputdlg({'Fid Thresh'},'Threshold for fiducials?',1,{'0.18'})));
    disp('Assume saturation recovery') % for now
    eval(['[',musID,'T1Img,',musID,'T1,', musID,'T1map,', musID,'T1mapError] = t1mapper(t1FN,numFids,t1Thresh,musID,''T1sat'');'])
    save(strcat(musID,'_T1map'), '-regexp',['^',musID,'T1'])
    cd(wd)
%     save
elseif get(handles.BrukimgTypeE,'Value')==1
    % load EPSI
    wd=cd;
    if strcmp(wd,'C:\Matlab\R2008b\work') || strcmp(wd,'C:\MATLAB\R2009a\work')
        CH_Ddrive= dir('D:\MRI_data\2009\RalphClinicalTNF\');
        if (numel(CH_Ddrive)~=0)
            cd('D:\mri_data\2009\')
        else
            CH_Qdrive= dir('q:\mri_data\2009\');
            if (numel(CH_Qdrive)~=0)
                cd('Q:\MRI_data\2009\')
            end
        end
    end
    [epsi_file,epsi_path]=uigetfile({'*.img;*.IMG','Raw data (*.img)';'*.*',...
        'All files (*.*)'},'Choose EPSI image :');
    if epsi_file==0 & epsi_path==0
        error('user canceled')
    end
    
    mySlash=regexp(epsi_path,'\');
    MusNameGs=epsi_path(mySlash(end-5)+1:mySlash(end-4)-1);
    myunbar=regexp(MusNameGs,'_');
    if ~isempty(myunbar) && numel(myunbar)==2, MusNameGs=MusNameGs([1:myunbar(1)-1,myunbar(2)+1:end]);end
    musID=[];
    while isempty(musID)
        musIDfn = inputdlg({'Mouse ID'},'Input Mouse ID',1,{MusNameGs});
        musIDfn = musIDfn{1};
        if isempty(musIDfn)
            warndlg('Mouse ID cannot be empty')
            musID=[];
            continue
        elseif isreal(str2double(musIDfn(1))) && ~(strcmpi(musIDfn(1),'j') || strcmpi(musIDfn(1),'i')) && ~isnan(str2double(musIDfn(1)))
            warndlg('Cannot have number as variable or start with a number')
            musID=[];
            continue
        elseif strfind(musIDfn,'-')
            warndlg('Mouse ID cannot have ''-''')
            musID=[];
            continue
        elseif strfind(musIDfn,' ')
            warndlg('Mouse ID cannot have ''a space''')
            musID=[];
            continue
        else
            musID=musIDfn;
        end
    end % while loop
    
    epsiFN = [epsi_path,epsi_file];
    epsi_imaFolder={[epsi_path(1:mySlash(end-4)) 'Matlab_ima']};
    epsi_imaFolder(2)={epsi_path(1:strfind(epsi_path,'\pdata\'))};
    epsiScanNmOrnt=epsi_file(strfind(epsi_file,'EPSImge')+7:strfind(epsi_file,'.img')-1);
    epsiSign = str2num(char(inputdlg({'Offset Sign'},'What is sign of offsets?',1,{'3'})));
    
%     [bay18d0epsi9AxImg,bay18d0epsi9Ax]=load_epsi([],'folderin','D:\MRI_data\2009\Stadler\bay18_07-27-09_d0\Matlab_ima','sign',3,{'FeetFirst','Prone'});
    eval(['[',musID,'epsi',epsiScanNmOrnt,'Img,',musID,'epsi',epsiScanNmOrnt,'] = load_epsi([],''folderin'',epsi_imaFolder,''sign'',epsiSign,myOptions.patOrient);']);
    if exist(strcat(musID,'_epsi.mat'),'file')
        save(strcat(musID,'_epsi'),'-append','-regexp',['^',musID,'epsi'])
    else
        save(strcat(musID,'_epsi'),'-regexp',['^',musID,'epsi'])
    end
    eval(['epsiImg=', musID,'epsi',epsiScanNmOrnt,'Img;'])
    [Nm,epsiHist]=hist(epsiImg(:),100);
%     iNm=cumtrapz(Nm(:))/sum(Nm(:));
%     mythresh = 0.72;
%     myWinLevel = epsiHist(find(iNm <= mythresh, 1, 'last' ));
%     myWinLevel = epsiHist(find(epsiHist >= mean(epsiImg(:))+std(epsiImg(:))*3.5,1,'first'));
    windowSize = 5;
    myDiff=diff(Nm)';
    myDiff=filter(ones(1,windowSize)/windowSize,1,cat(1,...
        repmat(myDiff(1),[windowSize 1]), myDiff, repmat(myDiff(end),[windowSize 1])));
    % remove padding
    myDiff([1:windowSize,end-windowSize+1:end])=[];
    negderivs = find(myDiff < 0);
    npeak = find(Nm == max(Nm(:)),1,'last');
    nfew = find(Nm <= 80 & Nm > 5);
    nfewright = nfew(find(nfew > npeak, 1,'last' ));
    myWinLevel = round(epsiHist(negderivs(find(negderivs > nfewright, 1 )))*100)/100;

    figure;hist(epsiImg(epsiImg~=0),100);
    ylim([0 800])
    otsuLev=myWinLevel;
    for x=1:size(epsiImg,3)
        otsuLev=min(otsuLev,graythresh(epsiImg(:,:,x)));
    end
    mysubimage(epsiImg,1,size(epsiImg,3),[],[],[0 myWinLevel],'copper');
    disp(strcat('load(''',musID,'_epsi'')'))
    disp(['mysubimage(',musID,'epsi',epsiScanNmOrnt,'Img,1,',num2str(size(epsiImg,3)),',[],[],[0 ',num2str(myWinLevel),'],''copper'');'])
    disp(['Otsu threshold ', num2str(otsuLev)])
    disp(['print -dpng -r200 ', musID,'epsi',epsiScanNmOrnt,'.png'])
    myfigOld = findall(0,'tag','MySubImage');
    eval(['print(max(myfigOld), ''-dpng'', ''-r200'', ''',musID,'epsi',epsiScanNmOrnt,'.png'')'])
%     eval(['print -dpng -r200 ', musID,'epsi',epsiScanNmOrnt,'.png'])
    
end  % end if which type of image/radiobutton

set(handles.LoadBrukerMRI_fig,'HandleVisibility','On')
set(handles.LoadBrukerMRI_fig,'Visible', 'On')
% clear radio button var in base
evalin('base','clear saddleCoil');
evalin('base','clear fixDoseYN');
drawnow;

% --- Executes on button press in readACQP.
function readACQP_Callback(hObject, eventdata, handles)
if get(handles.saddlecbx,'Value')==1
    saddleCoil='Yes';
else
    saddleCoil='No';
end
assignin('base','saddleCoil',saddleCoil);
readAcqpGUI

evalin('base','clear saddleCoil');

% --- Executes on button press in BrukimgTypeD.
function BrukimgTypeD_Callback(hObject, eventdata, handles)
set(handles.numSeriestxt,'Visible','On')
set(handles.numSeries,'Visible','On')
set(handles.musWt_txt,'Visible','On')
set(handles.musWt,'Visible','On')
% hObject    handle to BrukimgTypeD (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hint: get(hObject,'Value') returns toggle state of BrukimgTypeD

% --- Executes on button press in BrukimgTypeA.
function BrukimgTypeA_Callback(hObject, eventdata, handles)
set(handles.numSeriestxt,'Visible','Off')
set(handles.numSeries,'Visible','Off')
set(handles.musWt_txt,'Visible','Off')
set(handles.musWt,'Visible','Off')

% --- Executes when selected object is changed in OrientpanelHF.
function OrientpanelHF_SelectionChangeFcn(hObject, eventdata, handles)
% hObject    handle to the selected object in OrientpanelHF 
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% % read current values 1st
% % head, feet, prone, supine
% patOrient=[0 0 0 0];
% if get(handles.head1st,'Value')==1
%     patOrient(1)=1;
% else
%     patOrient(2)=1;
% end
% if get(handles.pronPos,'Value')==1
%     patOrient(3)=1;
% else
%     patOrient(4)=1;
% end
% switch hObject
%     case handles.head1st
%         set(handles.head1st,'Value',1)
%         set(handles.ft1st,'Value',patOrient(2))
%         set(handles.pronPos,'Value',patOrient(3))
%         set(handles.supPos,'Value',patOrient(4))
%     case handles.ft1st
%         set(handles.ft1st,'Value',1)
%         set(handles.head1st,'Value',patOrient(1))
%         set(handles.pronPos,'Value',patOrient(3))
%         set(handles.supPos,'Value',patOrient(4))
%     case handles.pronPos
%         set(handles.pronPos,'Value',1)
%         set(handles.head1st,'Value',patOrient(1))
%         set(handles.ft1st,'Value',patOrient(2))
%         set(handles.supPos,'Value',patOrient(4))
%     case handles.supPos
%         set(handles.supPos,'Value',1)
%         set(handles.head1st,'Value',patOrient(1))
%         set(handles.ft1st,'Value',patOrient(2))
%         set(handles.pronPos,'Value',patOrient(3))
% end

% --- Executes when selected object is changed in ImgTypepanel.
function ImgTypepanel_SelectionChangeFcn(hObject, eventdata, handles)
function Img2Save_Callback(hObject, eventdata, handles)
% --- Executes during object creation, after setting all properties.
function Img2Save_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
function numSeries_Callback(hObject, eventdata, handles)
% --- Executes during object creation, after setting all properties.
function numSeries_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
% --- Executes on button press in saddlecbx.
function saddlecbx_Callback(hObject, eventdata, handles)
% --- Executes on button press in hiGdcbx.
function hiGdcbx_Callback(hObject, eventdata, handles)



function musWt_Callback(hObject, eventdata, handles)
% hObject    handle to musWt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of musWt as text
%        str2double(get(hObject,'String')) returns contents of musWt as a double


% --- Executes during object creation, after setting all properties.
function musWt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to musWt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
