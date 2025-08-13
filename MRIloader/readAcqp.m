function acqpInfo=readAcqp(varargin)
% similar to IDL version of read_acqp
%
%sets file and folder names, assumes inputs in order: folder, file
%if only folder is given, does an early cd to start dialog at correct
%folder
% allow for anisotropic FOV CH 01-10-08

switch nargin
    case 0
        CH_Ddrive= dir('D:\mri_data\2009\');
        if (numel(CH_Ddrive)~=0)
            folder=uigetdir('D:\mri_data\2009\','Select Folder Containing ACQP File');
        else
            folder=uigetdir('X:\data\','Select Folder Containing ACQP File');
        end
    otherwise
        folder=varargin{1};
end
%makes sure folder ends with a \ or / (in case user enters, e.g., 'C:\data'
%instead of 'C:\data\'
j=strtrim(char(folder));
if length(j)<3
    return;
end
if (j(length(j))~='\' &  j(length(j))~='/')
    if (strfind(j,'\') ~= 0), j(length(j)+1)='\'; end;
    if (strfind(j,'/') ~= 0), j(length(j)+1)='/'; end;
    folder=j;
end

wd=cd;
cd(folder);
file='acqp';

fid=fopen(file);
if fid<=0
    error([folder ': Unable to open acqp file!']);
    acqpInfo='failed';
    %fclose(fid);
    return;
end
s1=fgetl(fid); s=s1;
while ~(strncmp(s1,'##END=',6))
    s1=fgetl(fid);
    s=strvcat(s,s1);
end
fclose(fid);
fullfile=[char(folder) char(file)];

%initialize structure
acqpInfo=struct('pulsProg',' ','acqProtocol',' ','txAtten',0,'receiverGain',0,...
    'repTime',0,'echoTime',0,'nEchoes',1,'recovTime',0,...
    'nSlices',0,'FOV',0,'matrix',zeros(1,3),'sliceThick',0,...
    'sliceSep',zeros(1,3),'sliceList',zeros(1,3),'sliceOffset',zeros(1,3),...
    'readOffset',0,'phaseOffset',0,...
    'nAverages',0,'imageType',' ','slicepackVec', zeros(1,3),...
    'nEvolutionCycles',0,'evolutionDelay',0,'flipAngle',0,'ideal90at',-1,'BFreq',400.0,...
    'SpecWidth',75000.0,'excitationPulse',0,'pulseShape','N/A','readOutDir','N/A',...
    'rareFactor',1,'fatSat','N/A','gating','N/A','wordType','N/A',...
    'rawWordType','N/A','byteOrder','N/A','saveTime',' ','filename',fullfile,...
    'musWt',25.0,'subjOrientHF','FeetFirst','subjOrientSP','Prone','PVver','5x');

%beginning of mass data crunching
line=strmatch('##$PULPROG=',s);
% add pulse program field because user can name protocol anything but pulse
% program is difficult to change.  CH 07-19-07
if ~isempty(line)
    j=s(line+1,:);
    j=strrep(j,'<',blanks(1));
    j=strrep(j,'>',blanks(1));
    j=strrep(j,'.ppg',blanks(1));    
    j=strtrim(j);
    acqpInfo.pulsProg=j;
end

line=strmatch('##$ACQ_protocol_name=',s);
if ~isempty(line)
    j=s(line+1,:);
    j=strrep(j,'<',blanks(1));
    j=strrep(j,'>',blanks(1));
    j=strtrim(j);
    acqpInfo.acqProtocol=j;
end

line=strmatch('##OWNER',s);
if ~isempty(line)
    j=s(line+1,:);
    %date line (after owner line) has $$ before date and a () expression after;
    %get rid of these
    a=strfind(j,'(');
%     j2=char(j);
    j2=j(4:a(1)-2);
    acqpInfo.saveTime=j2;
    % look at datenum, datestr, and datevec
end

line=strmatch('##$ACQ_repetition_time=',s);
if ~isempty(line)
    j=s(line+1,:);
    j2=str2num(strtrim(j));
    acqpInfo.repTime=j2;
end

line=strmatch('##$ACQ_recov_time',s);
if ~isempty(line)
    j=s(line+1,:);
    j2=str2num(strtrim(j));
    acqpInfo.recovTime=j2;
end

line=strmatch('##$ACQ_echo_time',s);
if ~isempty(line)
    j=s(line+1,:);
    j2=str2num(strtrim(j));
    acqpInfo.echoTime=j2;
end

line=strmatch('##$NA=',s);
if ~isempty(line)
    j=s(line,:);
    [j1 j2]=strread(j,'%s %f','delimiter','=');
    acqpInfo.nAverages=j2;
end

line=strmatch('##$ACQ_n_echo_images=',s);
if ~isempty(line)
    j=s(line,:);
    [j1 j2]=strread(j,'%s %f','delimiter','=');
    acqpInfo.nEchoes=j2;
end

line=strmatch('##$ACQ_fov',s);
if ~isempty(line)
    j=s(line+1,:);
    j1 = strread(j,'%f');
    %     if ndim==1
%     if numel(j1) > 1 && j1(1)==j1(2) && numel(j1) ~= 3
    if all(ismember(j1,j1(1))) %check if isotropic FOV
        acqpInfo.FOV=j1(1);
    else
        % changed to allow anisotropic FOV
        disp('non-isotropic FOV')
        acqpInfo.FOV=j1';
        % reverse FOV order because of RC vs XY CH 02-19-10
        if length(acqpInfo.FOV) > 1, acqpInfo.FOV=[acqpInfo.FOV(2) acqpInfo.FOV(1)]; end
    end
end

line=strmatch('##$NSLICES=',s);
if ~isempty(line)
    j=s(line,:);
    [j1 j2]=strread(j,'%s %f','delimiter','=');
    acqpInfo.nSlices=j2;
%     isempty(strfind(lower(acqpInfo.acqProtocol),'mapshim'))
end

line=strmatch('##$ACQ_slice_offset',s);
if ~isempty(line)
%     j=s(line+1,:);
    k=1;
    j1=[];
    j=s(line+k,:);
    while ~strncmp('##',j,2) && ~strncmp('$$',j,2)
        j2=strread(strtrim(j),'%f')';
        j1=[j1 j2];
        k=k+1;
        j=s(line+k,:);
    end
    acqpInfo.sliceOffset=j1;
    %     acqpInfo.sliceOffset=zeros(1,acqpInfo.nSlices);
    %     junk=strread(strtrim(j),'%f');
    %     if acqpInfo.nSlices > 1
    %         acqpInfo.sliceOffset(1:acqpInfo.nSlices)=junk(1:acqpInfo.nSlices);
    %     else
    %         acqpInfo.sliceOffset=junk(1);
    %     end
end
% Read direction offset CH 06-09-10
line=strmatch('##$ACQ_read_offset',s);
if ~isempty(line)
    j=s(line+1,:);
    % assume read offset is the same for all slices
    j1 = strread(strtrim(j),'%f');
    acqpInfo.readOffset=j1(1);
end
% Phase encod. direction offset CH 06-09-10
% for now ignore phase2_offset
line=strmatch('##$ACQ_phase1_offset',s);
if ~isempty(line)
    j=s(line+1,:);
    % assume read offset is the same for all slices
    j1 = strread(strtrim(j),'%f');
    acqpInfo.phaseOffset=j1(1);
end
line=strmatch('##$ACQ_slice_sepn=',s);
if ~isempty(line)
    j=s(line+1,:);
    [j1 j2] = strread(strtrim(j),'%f %f');
    acqpInfo.sliceSep=j1(1);
end

line=strmatch('##$ACQ_slice_thick',s);
if ~isempty(line)
    j=s(line,:);
    [j1 j2]=strread(j,'%s %f','delimiter','=');
    acqpInfo.sliceThick=j2;
end

line=strmatch('##$BYTORDA',s);
% raw data byte order
if ~isempty(line)
    j=s(line,:);
    [j1 j2]=strread(j,'%s %s','delimiter','=');
    acqpInfo.byteOrder=strtrim(char(j2));
end

% !!!!!!!!! imtype from reco can be different from saved FID
line=strmatch('##$GO_raw_data_format',s);
if ~isempty(line)
    j=s(line,:);
    [j1 j2]=strread(j,'%s %s','delimiter','=');

    test=strtrim(char(j2));
    % by default raw data is stored as 32-bit integer
    if strcmp(test,'GO_32BIT_SGN_INT')
        acqpInfo.rawWordType='int32';
    elseif strcmp(test,'GO_16BIT_SGN_INT')
        acqpInfo.rawWordType='int16';    
    elseif strcmp(test,'GO_32BIT_FLOAT')
        acqpInfo.rawWordType='float32';
    end
end

% for multi-echo, obj order is better to get from METHOD CH 11-13-08
% line=strmatch('##$ACQ_obj_order',s);
% if ~isempty(line)
%     j=s(line+1,:);
%     j1=str2num(strtrim(j));
%     acqpInfo.sliceList=j1;
% end

% gating
line=strmatch('##$ACQ_trigger_enable=',s);
if ~isempty(line)
    j=s(line,:);
    [j1 j2]=strread(j,'%s %s','delimiter','=');
    acqpInfo.gating=strtrim(char(j2));
end

% @_@_@_@_@_@_@_@_@_@_@_@_@_@_@_
%oldPV=0 --> PV 4
% oldPV=3 --> PV 3.0.2
% oldPV=1 --> PV 3.01 or PV 2
% @_@_@_@_@_@_@_@_@_@_@_@_@_@_@_

% find ##$ACQ_sw_version=( 65 ) line+1
% last PV on 4.7T was <PV 3.0.2pl1>, old is <PV 3.0.1>
oldPV=0;  %initialize to old paravision software is false
line=strmatch('##$ACQ_sw_version=',s);
if ~isempty(line)
    j=strtrim(s(line+1,:));
    acqpInfo.PVver=(j(2:end-1)); %remove < and >
    if (strcmp(acqpInfo.PVver,'PV 3.0.1')) | (strcmp(acqpInfo.PVver,'PV 2.Beta.1.12'))
        %(strcmp(j1,'<PV 3.0.2pl1>')) |
        % last ParaVision w/ 4.7T was 3.0.2pl1
        % PV 4 on 9.4T is <PV 4.0>
        oldPV=1;
    elseif (strcmp(acqpInfo.PVver,'PV 3.0.2pl1'))
        oldPV=3; % CH changed because new PV is v4.0
        % CH 07-19-07
        %elseif (strcmp(j1,'<<PV 3.0.2pl1>'))
    end    
end

line=strmatch('##$ACQ_grad_matrix',s);

if ~isempty(line)
    if oldPV==3  % CH changed because new PV is v4.0  CH 07-19-07                                        
        j=s(line+3,:);  %this works for new format, old format is line+1
        j1=fix(strread(j,'%f'));
        if strcmpi(acqpInfo.pulsProg, 'fmap_fq')
            j1=[inf inf inf];
        else
            j1=j1(2:4)'; %old format is j1(7:9)'
        end
    elseif oldPV==1  
        j=s(line+1,:);
        j1=strread(j,'%f');
        j1=j1(7:9)';
    elseif oldPV==0          
        j=s(line+1,:);
        k=1;
        j1=[];
%         j=s(line+k,:);
        while ~strncmp('##',j,2) && ~strncmp('$$',j,2)
%             j2=strread(j,'%f')';
            j1=[j1 strread(j,'%f')'];
            k=k+1;
            j=s(line+k,:);
        end  
        j1=j1(7:9);
    end
    % CH 3-5-07, if user changed angle
    if max(j1(:)) < 1
        j1(j1==max(j1(:)))=1;
        disp('Angle changed?')
    end
    acqpInfo.slicepackVec=j1;
    if isempty(strfind(acqpInfo.saveTime,'2004')) && oldPV~=1        
        saddleCoil='no';
%     elseif isempty(strfind(acqpInfo.saveTime,'2004')) && oldPV==0
%         saddleCoil='yes';
%         % even though saddle coil isn't used, PV 4.0 changed the
%         % configuration that matches as if the saddle coil was used.
%         % CH 07-19-07
    else
        % image is old enough to be saddle coil
        % CH 03-02-07
        disp('**** new birdcage tried around April 22, 2005 ***')
        % army41d17 has photo of cast, so should be birdcage.
        vars = evalin('base','who');
        if ismember('saddleCoil',vars)
            saddleCoil = evalin('base', 'saddleCoil');
        else
            saddleCoil = questdlg('Was saddle coil used?','Saddle Coil Fix');
            assignin('base','saddleCoil',saddleCoil);
        end
        clear vars
    end % if old/saddle coil
    switch lower(saddleCoil)
        case 'yes'
            % acqpInfo.SaddleCoil='Yes';
            % saddle coil, axial is sagittal & vice versa
            if j1(1)==1
                acqpInfo.imageType='Axial';
            elseif j1(2)==1
                acqpInfo.imageType='Coronal';
            elseif j1(3)==1
                acqpInfo.imageType='Sagittal';
            end
        case 'no'
            if j1(1)==1
                acqpInfo.imageType='Sagittal';
            elseif j1(2)==1
                acqpInfo.imageType='Coronal';
            elseif j1(3)==1
                acqpInfo.imageType='Axial';
            end
    end % case
end

line=strmatch('##$ACQ_rare_factor=',s);
if ~isempty(line)
    j=s(line,:);
    [j1 j2]=strread(j,'%s %f','delimiter','=');
    acqpInfo.rareFactor=j2;
end

%rec gain
line=strmatch('##$RG=',s);
if ~isempty(line)
    j=s(line,:);
    [j1 j2]=strread(j,'%s %f','delimiter','=');
    acqpInfo.receiverGain=j2;
end
%attenuators
line=strmatch('##$TPQQ=',s);
if ~isempty(line)
    j=s(line+1,:);
    a=strfind(j,',');
    j2=char(j);
    acqpInfo.txAtten=[str2num(j2(a(1)+1:a(2)-1)) str2num(j2(a(3)+1:a(4)-1))];
end

line=strmatch('##$ACQ_flip_angle=',s);
if ~isempty(line)
    j=s(line,:);
    [j1 j2]=strread(j,'%s %f','delimiter','=');
    acqpInfo.flipAngle=j2;
end

line=strmatch('##$NR=',s);
if ~isempty(line)
    j=s(line,:);
    [j1 j2]=strread(j,'%s %f','delimiter','=');
    acqpInfo.nEvolutionCycles=j2;
end

line=strmatch('##$BF1=',s);
if ~isempty(line)
    j=s(line,:);
    [j1 j2]=strread(j,'%s %f','delimiter','=');
    fprintf(1, 'Basic Freq.: %3.7f \n',j2);
    acqpInfo.BFreq=j2;
end

%bandwidth
line=strmatch('##$SW_h=',s);
if ~isempty(line)
    j=s(line,:);
    [j1 j2]=strread(j,'%s %f','delimiter','=');
    acqpInfo.SpecWidth=j2;
end

fid=fopen('method');
if fid<=0
    %evolution delay is in another file
    disp('Unable to open method file!')
    ms=[];
else
    ms1=fgetl(fid); ms=ms1;
    while ~(strncmp(ms1,'##END=',6))
        ms1=fgetl(fid);
        ms=strvcat(ms,ms1);
    end
    fclose(fid);
end

% % need method file to fix anti-alias -> matrix from ACQP
% line=strmatch('##$PVM_AntiAlias',ms);
% if ~isempty(line+1)
%     AA=ms(line+1,:);    
% end

line=strmatch('##$ACQ_dim=',s);
if ~isempty(line)
    j=s(line,:);
    [qq , j2]=strread(j,'%s %f','delimiter','=');
    ndim=j2;
    % use method file to get matrix because it know about anti-alias tricks and
    % if the matrix is intentionally anisotropic e.g., sagittal 128x256.  CH
    % 06-09-10
    line=strmatch('##$PVM_Matrix=',ms);
    if ~isempty(line+1)
        j=ms(line+1,:);
        if ndim==2
        [j1 j2]=strread(j,'%f %f ');
        % reverse order, RC vs XY
        acqpInfo.matrix=[j2 j1 1];
            %         line=strmatch('##$ACQ_size=',s);
            %         if ~isempty(line)
            %             j=s(line+1,:);
            %             [j1 j2]=strread(j,'%f %f ');
            %             if strfind(AA,'2') % anti-aliasing on
            %                 j1=j1/4;
            %             elseif j1/2~=j2 && isempty(strfind(lower(acqpInfo.acqProtocol),'epsi'))
            %                 %acqProtocol:1-TriPilot
            %                 disp(acqpInfo.acqProtocol)
            %                 disp(['ACQP has ',j])
            %                 prompt = {'What is the 1st dim size?', 'What is the 2nd dim size?'};
            %                 dlg_title = 'Matrix Size?';
            %                 num_lines= 2;
            %                 % reverse order, RC vs XY  - CH 03-20-10
            %                 defAns={num2str(j2),num2str(j1/4)};
            %                 newmatrixsz = str2num(char(inputdlg(prompt,dlg_title,num_lines,defAns)));
            %                 j1=newmatrixsz(1); j2=newmatrixsz(2);
            %             else
            %                 j1=j1/2;
            %             end
            %             acqpInfo.matrix=[j1 j2 1];
            %         end
        elseif ndim==3
            [j1 j2 j3]=strread(j,'%f %f  %f');
            % reverse order, RC vs XY
            acqpInfo.matrix=[j2 j1 j3];
%             line=strmatch('##$ACQ_size=',s);
%             if ~isempty(line)
%                 j=s(line+1,:);
%                 [j1 j2 j3]=strread(j,'%f %f %f');
%                 if j1/2~=j2
%                     prompt = {'What is the 1st dim size?', 'What is the 2nd dim size?'};
%                     dlg_title = 'Matrix Size?';
%                     num_lines= 2;
%                     % reverse order, RC vs XY  - CH 03-20-10
%                     defAns={num2str(j2),num2str(j1/4)};
%                     newmatrixsz = str2num(char(inputdlg(prompt,dlg_title,num_lines,defAns)));
%                     j1=newmatrixsz(1); j2=newmatrixsz(2);
%                 else
%                     j1=j1/2;
%                 end
%                 acqpInfo.matrix=[j1 j2 j3];
%             end
        elseif ndim==1
%             line=strmatch('##$ACQ_size=',s);
%             if ~isempty(line)
%                 j=s(line+1,:);
%                 [j1]=strread(j,'%f');
%                 acqpInfo.matrix=[j1(1)/2 1 1];
%             end
            j1 = strread(j,'%f');            
            acqpInfo.matrix=[j1 1 1];
        end
    end % if pvm_matrix line exist
end
% done with matrix

if (acqpInfo.nEvolutionCycles > 1 && fid > 0)
    if oldPV==1
        line=strmatch('##$ACQ_vd_list=',s);
        if ~isempty(line)
            j=s(line+1,:);
            j1=strread(j,'%f');
            acqpInfo.evolutionDelay=j1(1);
        end
    else
        if oldPV==3
            line=strmatch('##$FLASHVD_VarDelay=',ms);
            if ~isempty(line)
                j=ms(line,:);
                [j1 j2]=strread(j,'%s %f','delimiter','=');
                acqpInfo.evolutionDelay=j2(1)/1000;
            end
        elseif  oldPV==0
            line=strmatch('##$PVM_EvolutionDelay=',ms);
            if ~isempty(line)
                j=ms(line,:);
                [j1 j2]=strread(j,'%s %f','delimiter','=');
                acqpInfo.evolutionDelay=j2(1);  % new dynamic in sec not msec
            end
        end
        % for EPSI data
        epsi=strmatch('##$Method=ePSI',ms);
        if ~isempty(epsi)
            line=strmatch('##$EchoTime=',ms);
            j=ms(line,:);
            [j1 j2]=strread(j,'%s %f','delimiter','=');
            acqpInfo.echoTime=j2;
            acqpInfo.nEchoes=acqpInfo.matrix(2);
            acqpInfo.matrix(2)=acqpInfo.nEvolutionCycles;
            acqpInfo.nEvolutionCycles=1;
            disp('fixed EPSI params')
        end
    end %if oldPV
end %end if nEvCyl > 1 and method exist

if fid > 0  % method file exist
    %get excitation pulse
    line=strmatch('##$ExcPulse=(',ms);
    if ~isempty(line)
        j=ms(line,:);
        J=strread(j,'%s','delimiter','=(');
        % need to deal with ( and avoid LIB_EXCITATION
        %     j1=strread(j1{3}(1:12),'3.2%f','delimiter',',');
        j1=strread(J{3},'%f',1,'delimiter',',');
        acqpInfo.excitationPulse=j1(1);
    end
    % get attenuation for ideal 90 deg FA
    line=strmatch('##$PVM_RefAttCh1=',ms);
    if ~isempty(line)
        j=ms(line,:);        
        j1=strread(j,'%s','delimiter','=');
        acqpInfo.ideal90at=strtrim(j1{2});
    end
    %get readout direction
    line=strmatch('##$PVM_SPackArrReadOrient=',ms);
    if ~isempty(line)
        j=ms(line+1,:);
        J=strread(j,'%s'); % can be more than one dir (tripilot)
        acqpInfo.readOutDir=J{1};
    end   
    %get fat sup on/off
    line=strmatch('##$PVM_FatSupOnOff=',ms);
    if ~isempty(line)
        j=ms(line,:);
        j1=strread(j,'%s','delimiter','=');
        acqpInfo.fatSat=strtrim(j1{2});
    end
    % RF pulse shape
    line=strmatch('##$ExcPulseEnum=',ms);
    if ~isempty(line)
        j=ms(line,:);
        j1=strread(j,'%s','delimiter','=');
        acqpInfo.pulseShape=strtrim(j1{2});
    end
    % slice list (object order)
    line=strmatch('##$PVM_ObjOrderList=',ms);
    if ~isempty(line)
        j=ms(line+1,:);
        j1=str2num(strtrim(j));
        acqpInfo.sliceList=j1;
    end
    % DW B values
    if ~isempty(strfind(acqpInfo.pulsProg,'DtiStandard'))
        line=strmatch('##$PVM_DwBvalEach=',ms);
        if ~isempty(line)
            j=ms(line+1,:);
            j1=str2num(strtrim(j));
            acqpInfo.dwiBval=j1;
        end
    else
        acqpInfo.dwiBval='NA';
    end
else
    acqpInfo.excitationPulse=0;
    acqpInfo.readOutDir='unk';
    acqpInfo.fatSat='unk';
    acqpInfo.pulseShape='unk';
    acqpInfo.sliceList='unk';
end
            
%get reconstruction info
%recofolder=uigetdir(strcat(folder,'pdata\1\'),'Select Reco Directory');
if (strfind(folder,'\') ~= 0), recofolder=strcat(folder,'pdata\1\'); end;
if (strfind(folder,'/') ~= 0), recofolder=strcat(folder,'pdata/1/'); end;
cd(recofolder);

fid=fopen('reco');
if fid<=0
    disp('Unable to open reco file!')
    acqpInfo='failed';
    return;
end
rs1=fgetl(fid); rs=rs1;
while ~(strncmp(rs1,'##END=',6))
    rs1=fgetl(fid);
    rs=strvcat(rs,rs1);
end
fclose(fid);
% !!!!!!!!! imtype from reco can be different from saved FID
line=strmatch('##$RECO_wordtype=',rs);
if ~isempty(line)
    j=rs(line,:);
    [j1 j2]=strread(j,'%s %s','delimiter','=');

    test=strtrim(char(j2));
    if strcmp(test,'_32BIT_SGN_INT')
        acqpInfo.wordType='int32';
    elseif strcmp(test,'_16BIT_SGN_INT')
        acqpInfo.wordType='int16';
    elseif strcmp(test,'_8BIT_UNSGN_INT')
        acqpInfo.wordType='uint8';
    elseif strcmp(test,'_32BIT_FLOAT')
        acqpInfo.wordType='float32';
    end
end

%get sample physical orientation as told to ParaVision
cd(folder);
cd('..')  %  go up one level to find subject file
fid=fopen('subject');
if fid<=0
    disp('Unable to open subject file!')
    acqpInfo='failed';
    return;
end
ss1=fgetl(fid); ss=ss1;
while ~(strncmp(ss1,'##END=',6))
    ss1=fgetl(fid);
    ss=strvcat(ss,ss1);
end
fclose(fid);

line=strmatch('##$SUBJECT_entry=SUBJ_ENTRY_',ss);
if ~isempty(line)
    j=ss(line,:);
    k=strfind(j,'_');
    acqpInfo.subjOrientHF=deblank(j(k(3)+1:end));
end

line=strmatch('##$SUBJECT_position=SUBJ_POS_',ss);
if ~isempty(line)
    j=ss(line,:);
    k=strfind(j,'_');
    acqpInfo.subjOrientSP=deblank(j(k(3)+1:end));
end

line=strmatch('##$SUBJECT_name_string',ss);
if ~isempty(line+1)
    j=ss(line+1,:);
    k1=strfind(j,'<');
    k2=strfind(j,'>');
    disp(['ID in PV is: ', ss(line+1,k1+1:k2-1)])
end

line=strmatch('##$SUBJECT_weight=',ss);
if ~isempty(line)
    j=ss(line,:);
    k=strfind(j,'=');
    acqpInfo.musWt=str2double(deblank(j(k(end)+1:end)))*1e3;
    disp(['Mouse wt. ' num2str(acqpInfo.musWt)])
end

% read IMND file for old data
% added 3-24-07 CH
% changed to 2005 because IMND methods still used, 4-18-07 CH
cd(folder);
if (~isempty(strfind(acqpInfo.saveTime,'2005')) || ~isempty(strfind(acqpInfo.saveTime,'2004'))) && exist('imnd','file')
    fid=fopen('imnd');

    if fid<=0
        error([folder ': Unable to open IMND file!']);
    else
        s1=fgetl(fid); IMND=s1;
        while ~(strncmp(s1,'##END=',6))
            s1=fgetl(fid);
            IMND=strvcat(IMND,s1);
        end
        fclose(fid); clear s1

        line=strmatch('##$IMND_evolution_delay',IMND);
        if ~isempty(line)
            j=IMND(line+1,:);
            j1=strread(j,'%f');
            acqpInfo.evolutionDelay=j1(1);
        end
        line=strmatch('##$IMND_n_echo_images',IMND);
        if ~isempty(line) %length(line)~=0
            j=IMND(line,:);
            [j1 j2]=strread(j,'%s %f','delimiter','=');
            acqpInfo.nEchoes=j2;
        end
        %get excitation pulse
        line=strmatch('##$IMND_pulse_length=',IMND);
        if ~isempty(line)
            j=IMND(line,:);
            [j1 j2]=strread(j,'%s %f','delimiter','=');
            acqpInfo.excitationPulse=j2(1)/1000; %old is in us?
        end
        disp('ExcPuls/nEcho/evolutionDelay from IMND')
    end % end if fid
end % end if 2004
cd(wd)  %return to working dir