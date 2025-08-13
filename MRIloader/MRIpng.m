function MRIpng(varargin)

%this program will make png files for each MRI image acquired
%and save the "anatomical" MRI images that you specify.
%seList is an array containing the spin echo images to save
%Specify them by the actual folder number(s).
%It ignores DCE so use makeDCE to look at DCE images.
% You can pass the mouse ID as keyword 'mus' followed by the mouse ID
% and the parent folder as keyword 'folderin' followed by the parent folder
% changed to look for the actual folder number rather than having to count
% the folders you want CH 3-22-07
% allow for anisotropic FOV CH 01-10-08

seList=varargin{1};
musID = 0;
folder = 0;
for i=1:nargin %Look for keywords
    if (strcmpi(varargin{i}, 'mus'))
        musID = varargin{i+1};
    elseif (strcmpi(varargin{i}, 'folderin'))
        parfolder = varargin{i+1};
    elseif (strcmpi(varargin{i}, 'patOrient'))
        patOrient = varargin{i+1};
    end %of if stuff
end %of while

info=[];
info.n=0;
info.folders={[]};
seSaved=0;

wd=cd;
if folder == 0
    CH_Ddrive= dir('D:\mri_data\2012\');
    if (numel(CH_Ddrive)~=0)
        if strcmp(wd,'C:\MATLAB\R2011b\work') || strcmp(wd,'C:\MATLAB\R2010b\work')
            parfolder=uigetdir('D:\mri_data\2012\','Select Parent Folder');
        else
            parfolder=uigetdir(wd,'Select Parent Folder');
        end
    else
        parfolder=uigetdir('V:\data\','Select Parent Folder');
    end
end

if musID == 0
    mySlash=regexp(parfolder,'\');
    MusNameGs=parfolder(mySlash(end-1)+1:mySlash(end)-1);
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
end % if musID ==0, i.e., not defined yet

j=char(parfolder);
if (j(length(j))~='\' && j(length(j))~='/')
    j(length(j)+1)='\';
    parfolder=j;
end
dirs=struct2cell(dir(parfolder));
dirs=char(dirs{1,:});
% Folders must start with a digit; only use correct ones
good=find(dirs(:,1)>='0' & dirs(:,1)<='9');
goodsize=size(good);
if ~goodsize(1)
    errordlg('Illegal parent directory!','Error!');
    return;
end
dirs=cellstr(dirs(good,:));
dirs=strcat(parfolder,dirs);
%sort directories
sortDir=zeros(size(dirs,1),1);
for n=1:size(dirs,1)
    testdir=dirs{n,1};
    findNum=strfind(testdir,'\');
    sortDir(n)=str2num(testdir(findNum(end)+1:end));
end
[sortDr, idx] = sort(sortDir);
sortDirs=dirs(idx(:,1),1);
[zzz, idxSv, zz]=intersect(sortDr,seList); clear junk
for w=1:length(sortDirs)
    if sum(strcmp(sortDirs(w),info.folders))==0
        info.folders{info.n+1}=char(sortDirs(w));
        info.n=info.n+1;
        cd(strcat(sortDirs{w},'\pdata\1\'));
        test4file=dir('*.img');
        if ~isempty(test4file)
            imgPos=strfind(test4file.name,'.img');
            svNm=test4file.name(1:imgPos-1);
            % 03-03-07 read ACQP first, then skip read2dseq if EPSI or DCE
            acqpInfo=readAcqp(strcat(sortDirs{w}));
            if size(acqpInfo.FOV)==1, acqpInfo.FOV(2)=acqpInfo.FOV(1); end
            acqpInfo.FOV=acqpInfo.FOV*10;
            imgsize=acqpInfo.matrix;
            if imgsize(3)==1
                % changed index to match 17 Jan 2012 CH
                acqpInfo.zscale=(acqpInfo.sliceSep/(min([acqpInfo.FOV(1)/acqpInfo.matrix(1) acqpInfo.FOV(2)/acqpInfo.matrix(2)])));
                % stack of 2D slices
            else
                % true 3D image
                acqpInfo.zscale=(1/(min(acqpInfo.FOV)/acqpInfo.matrix(3)));
            end
            if (isempty(strfind(lower(acqpInfo.acqProtocol),'epsi')) || isempty(strfind(lower(acqpInfo.pulsProg),'press'))) && acqpInfo.nEvolutionCycles <=1
                % return acqpInfo as read2dseq will correct head first
                % error
                if isempty(strfind(acqpInfo.PVver,'PV 5.'))
                    [Img,acqpInfo] = read2dseq(strcat(sortDirs{w},'\pdata\1\',test4file.name),[],patOrient);
                else
                    % don't need acqpinfo again if using PV 5
                    Img = read2dseq(strcat(sortDirs{w},'\pdata\1\',test4file.name),[],patOrient);
                end
            else
                disp('EPSI, DCE, or 4D, skipped')
                continue
            end
            cd(sortDirs{w})
            cd('..')
            cd('..')
            
            if ismember(info.n,idxSv) %find(info.n==seList)>0
                eval([svNm, 'Img =Img;'])
                eval([svNm, '=acqpInfo;'])  %zscale added by read2dseq
                if length(acqpInfo.FOV)==1, acqpInfo.FOV(2)=acqpInfo.FOV(1); end
                % changed index to match 17 Jan 2012 CH
                eval(strcat(svNm,'.scale=[acqpInfo.FOV(1)/acqpInfo.matrix(1) acqpInfo.FOV(2)/acqpInfo.matrix(2) acqpInfo.sliceSep];'));
                if seSaved==0
                    eval(['save ',musID,'_se.mat ', svNm, 'Img ', svNm])
                    seSaved=1;
                else
                    eval(['save -append ',musID,'_se.mat ', svNm, 'Img ', svNm])
                end
                disp([svNm, ' saved'])
            end  %end if n is in save list
            if acqpInfo.nEvolutionCycles <=1 && acqpInfo.nSlices ~= 0 %press
                
                if acqpInfo.nSlices > 3
                    xw=min([ceil(sqrt(acqpInfo.nSlices)) ceil(acqpInfo.nSlices/ceil(sqrt(acqpInfo.nSlices)))]);
                    yw=max([ceil(sqrt(acqpInfo.nSlices)) ceil(acqpInfo.nSlices/ceil(sqrt(acqpInfo.nSlices)))]);
                else
                    xw=1;
                    yw=acqpInfo.nSlices;
                    if acqpInfo.nSlices == 1 && acqpInfo.matrix(3) > 1
                        disp('Image is 3D')
                        disp(acqpInfo.acqProtocol)
                        % fix matrix(3) not nSlices
                        xw=min([ceil(sqrt(acqpInfo.matrix(3))) ceil(acqpInfo.matrix(3)/ceil(sqrt(acqpInfo.matrix(3))))]);
                        yw=max([ceil(sqrt(acqpInfo.matrix(3))) ceil(acqpInfo.matrix(3)/ceil(sqrt(acqpInfo.matrix(3))))]);
                    end
                end
                if acqpInfo.nEchoes > 1 && isempty(strfind(lower(acqpInfo.acqProtocol),'mapshim'))
                    disp('Reshaping multi-echo image')
                    %                     Img=squeeze(Img(:,:,round(acqpInfo.nEchoes/2),:));
                    Img=reshape(Img,[acqpInfo.matrix(2:-1:1),acqpInfo.nEchoes*acqpInfo.nSlices]);
                    xw=min([ceil(sqrt(acqpInfo.nSlices*acqpInfo.nEchoes))...
                        ceil(acqpInfo.nSlices*acqpInfo.nEchoes/ceil(sqrt(acqpInfo.nSlices*acqpInfo.nEchoes)))]);
                    yw=max([ceil(sqrt(acqpInfo.nSlices*acqpInfo.nEchoes))...
                        ceil(acqpInfo.nSlices*acqpInfo.nEchoes/ceil(sqrt(acqpInfo.nSlices*acqpInfo.nEchoes)))]);
                elseif ~isempty(strfind(lower(acqpInfo.acqProtocol),'mapshim'))
                    % fix mapshim because it has phase image followed by
                    % mag
                    xw=min([ceil(sqrt(acqpInfo.matrix(1)*acqpInfo.nEchoes))...
                        ceil(acqpInfo.matrix(1)*acqpInfo.nEchoes/ceil(sqrt(acqpInfo.matrix(1)*acqpInfo.nEchoes)))]);
                    yw=max([ceil(sqrt(acqpInfo.matrix(1)*acqpInfo.nEchoes))...
                        ceil(acqpInfo.matrix(1)*acqpInfo.nEchoes/ceil(sqrt(acqpInfo.matrix(1)*acqpInfo.nEchoes)))]);
                end
                mysubimage(Img,xw,yw,[],[],[],'bone');
                %mysubimage (img,r,c,gaps,sgaps,mylim,mycmap,mytitle)
                footer(strcat(sortDirs{w}, svNm));
                myfig = findall(0,'tag','MySubImage');
%                 eval(['print -f',num2str(max(myfig)),' -dpng -r180 ',svNm,'.png'])
                eval(['print -dpng -r180 ',svNm,'.png'])
            end %end if not DCE
        else
            disp(['Img file not exist in ' num2str(w)]);
        end   %end if  test4file
    end  %end if sortDirs, info.folder check
end  %end for
cd(wd);
disp('Finished making MRI png pictures')