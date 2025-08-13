function varargout = read2dseq(brukerfile,varargin);
% Written by C. Haney Sept. 2005
% Reads in reconstructed MRI image stored by Bruker ParaVision in the 2dseq
% file.  Assumes 1st argument is 2dseq file with path or renamed 2dseq IMG
% file by readAcqpInfo_GUI with full path.  2nd argument should be
% acqpInfo.  If acqpInfo is omitted it will be created.  The third output
% variable is optional for 3D (isotropic) images.
% Modified Feb 2010, PV 4 and PV 5 do not need permute or any flipping as
% long as you leave the default, headFirst/prone, during acquisition.
% Paravision will flip things around if the user changes the defaults.  See
% bay16DeadLeg MRI and microCT
%
% example:
% [mriImg8, acqpInfo8] =
% read2dseq('D:\m61_09-19-05_d0\Pel091905.xJ1\8\pdata\1\army61d0.img',[],{'HeadFirst','Supine'});
% brukerflie is always 1st arg, acqpinfo is 2nd, and HeadFirst is 3rd
% brukerfile = varargin{1}
pdataPos=strfind(brukerfile,'pdata\1\');
brukerfolder=brukerfile(1:pdataPos-1);
if nargin == 2 && isstruct(varargin{1})
    acqpInfo=varargin{1};
else
    acqpInfo=readAcqp(brukerfolder);
    acqpInfo.FOV=acqpInfo.FOV*10;   % FOV in mm   !!!!!!!!
end
fixFt1st = {'HeadFirst','Supine'};%{'FeetFirst','Prone'};
if nargin == 3
    % give option to not fix assumed feet 1st, CH 09-11-07  
    fixFt1st = varargin{2};
end
imtype=acqpInfo.wordType;
% fix RC vs XY 17 Jan 2012
imgsize = [acqpInfo.matrix(2) acqpInfo.matrix(1) acqpInfo.matrix(3)];
img3D=1;
if imgsize(3)==1
    imgsize(3)=acqpInfo.nSlices;
    if acqpInfo.nEvolutionCycles > 1
        imgsize(4)=acqpInfo.nEvolutionCycles;
    end
    if acqpInfo.nEchoes > 1
        imgsize(3)=acqpInfo.nEchoes;
        imgsize(4)=acqpInfo.nSlices;
    end
    acqpInfo.zscale=(acqpInfo.sliceSep/(min(acqpInfo.FOV)/acqpInfo.matrix(1)));
    img3D=0;  % stack of 2D slices
else
    % true 3D image
    acqpInfo.zscale=(1/(min(acqpInfo.FOV)/acqpInfo.matrix(3)));
end
if ~isempty(strfind(lower(acqpInfo.acqProtocol),'mapshim'))
    % fix mapshim because it has phase image followed by mag
    imgsize(3)=imgsize(3)*2;    
end

if ~isempty(strfind(acqpInfo.pulsProg,'DtiStandard'))
    % fix DWI/DTI
    imgsize(3)=imgsize(3)*length(acqpInfo.dwiBval);    
end
fid=fopen(brukerfile);

if (fid <= 0)
    %fprintf('couldn''t open first file! %s\n', brukerfile);
    error([brukerfile ': can''t open file for reading']);
    return
else

    if (brukerfile ~= 0)
        fprintf(1, 'processing %s\n', brukerfile);
        %fid=fopen(brukerfile);
        if (fid ~= 0)
            mriImgin = fread(fid, imtype);
            if  acqpInfo.nSlices ~= 0 % PRESS sepctrum
                %reshape data into m x n, and transpose x and y. 
                %This works if ParaVision is told Feetfirst and Prone.
                if acqpInfo.nSlices == 1 && img3D==0 && acqpInfo.nEvolutionCycles == 1 && acqpInfo.nEchoes == 1% image is not 3D                    
                    if strcmp(acqpInfo.imageType,'Axial') && (strcmp(acqpInfo.subjOrientHF,'FeetFirst') || strcmp(fixFt1st{1},'FeetFirst'))
                        % permute if ACQP is feet 1st    
                        if isempty(strfind(acqpInfo.PVver,'PV 5'))
                            disp('___ Not Head 1st!!! ____')
                            disp(['ACQP ', acqpInfo.subjOrientHF, 'User says ',fixFt1st{1}])
                        end
                        mriImg=permute(reshape(mriImgin,squeeze(imgsize)),[2 1]);
                    else % not axial or HeadFirst
                        % this should be the default after Feb 1, 2010,
                        % using PV 5
                        mriImg=reshape(mriImgin,squeeze(imgsize));
%                         if length(acqpInfo.FOV) > 1, acqpInfo.FOV=[acqpInfo.FOV(2) acqpInfo.FOV(1)]; end
                    end
                elseif (length(imgsize) == 3 && acqpInfo.nSlices~=1) || (length(imgsize) == 3 && img3D==1)                    
                    if strcmp(acqpInfo.imageType,'Axial') && (strcmp(acqpInfo.subjOrientHF,'FeetFirst') || strcmp(fixFt1st{1},'FeetFirst'))
                        % permute if ACQP is feet 1st or user says should be
                        % feet 1st, CH 06-11-08
                        if isempty(strfind(acqpInfo.PVver,'PV 5'))
                            disp('___ Not Head 1st!!! ____')
                            disp(['ACQP ', acqpInfo.subjOrientHF, 'User says ',fixFt1st{1}])
                        end
                        mriImg=permute(reshape(mriImgin,squeeze(imgsize)),[2 1 3]);
                    else % not axial or HeadFirst was FeetFirst 12-04-07
                        % this should be the default after Feb 1, 2010,
                        % using PV 5
                        mriImg=reshape(mriImgin,squeeze(imgsize));                        
%                         if length(acqpInfo.FOV) > 1, acqpInfo.FOV=[acqpInfo.FOV(2) acqpInfo.FOV(1)]; end
                    end
                elseif (length(imgsize) == 4)                    
                    if strcmp(acqpInfo.imageType,'Axial') && (strcmp(acqpInfo.subjOrientHF,'FeetFirst') || strcmp(fixFt1st{1},'FeetFirst'))
                        % permute if ACQP is feet 1st or user says should be
                        % feet 1st, CH 06-11-08
                        if isempty(strfind(acqpInfo.PVver,'PV 5'))
                            disp('___ Not Head 1st!!! ____')
                            disp(['ACQP ', acqpInfo.subjOrientHF, 'User says ',fixFt1st{1}])
                        end
                        mriImg=permute(reshape(mriImgin,squeeze(imgsize)),[2 1 3 4]);
                    else % not axial or HeadFirst was FeetFirst 12-04-07
                        % this should be the default after Feb 1, 2010,
                        % using PV 5
                        mriImg=reshape(mriImgin,squeeze(imgsize));                        
%                         if length(acqpInfo.FOV) > 1, acqpInfo.FOV=[acqpInfo.FOV(2) acqpInfo.FOV(1)]; end
                    end
                end
                fclose(fid);
            end
        end
    end
     if strcmp(acqpInfo.imageType,'Axial') && strcmp(acqpInfo.subjOrientHF,...
             'HeadFirst') && strcmp(fixFt1st{1},'FeetFirst')  && isempty(strfind(lower(acqpInfo.acqProtocol),'tripilot'))
                    % permute if ACQP is feet 1st or user says should be
                    % feet 1st, CH 06-11-08 but skip if tripilot CH 6-21-08
        if (length(imgsize) == 3) && acqpInfo.nSlices ~= 0 % PRESS sepctrum
            mriImg=mriImg(:,:,end:-1:1); % reverse slices to be foot first
        elseif (length(imgsize) == 4) && acqpInfo.nSlices ~= 0 % PRESS sepctrum
            mriImg=mriImg(:,:,end:-1:1,:); % reverse slices to be foot first
        end
        % reverse sliceOffset too.
        acqpInfo.sliceOffset=acqpInfo.sliceOffset(end:-1:1);
        acqpInfo.sliceList=acqpInfo.sliceList(end:-1:1);
        disp('Was subject really imaged head 1st')
        disp('Assuming subject was foot 1st and reversing order')
     end
     if strcmp(acqpInfo.imageType,'Sagittal') && (strcmp(acqpInfo.subjOrientHF,'FeetFirst') || strcmp(fixFt1st{1},'FeetFirst'))
                    % permute if ACQP is feet 1st or user says should be
                    % feet 1st, CH 06-11-08
        if (length(imgsize) == 3)
            mriImg=mriImg(:,:,end:-1:1); % flip left/right
        elseif (length(imgsize) == 4)
            mriImg=mriImg(:,:,end:-1:1,:); % flip left/right
        end
        % reverse sliceOffset too.
        acqpInfo.sliceOffset=acqpInfo.sliceOffset(end:-1:1);
        acqpInfo.sliceList=acqpInfo.sliceList(end:-1:1);
    end
    if strcmp(acqpInfo.imageType,'Coronal') && strcmp(acqpInfo.subjOrientSP,'Supine') && ~strcmp(fixFt1st{2},'Supine')
        mriImg=flipdim(mriImg,1);        
        disp('Was subject really imaged supine?')
        disp('Assuming subject was prone; correcting coronal image')
        disp(['ACQP ', acqpInfo.subjOrientSP, 'User says ',fixFt1st{2}])
    end
    % rotate axial/sagittal slices 90 deg CH 02-07-10, PV 5    
    if ~isempty(strfind(acqpInfo.PVver,'PV 5'))
        if ~strcmp(acqpInfo.imageType,'Coronal')  & (acqpInfo.FOV ~= 0 & acqpInfo.sliceThick ~= 0) & exist('mriImg','var')
            % exclude spectrum (1D)
            if (length(imgsize) == 3)
                mrirot = zeros([size(mriImg,2),size(mriImg,1),size(mriImg,3)]);
            elseif (length(imgsize) == 4)
                mrirot = zeros([size(mriImg,2),size(mriImg,1),size(mriImg,3),size(mriImg,4)]);
            end
            % don't use acqpInfo.nSlices because mapshim is 128 but nSlice
            % = 1, DCE and EPSI are 4D so can't use imgsize
            if ~isempty(strfind(lower(acqpInfo.acqProtocol),'mapshim')),NumSlices=imgsize(3); else NumSlices=acqpInfo.nSlices; end
            for nslice=1:NumSlices
                if (length(imgsize) == 3)
                    mrirot(:,:,nslice)=rot90(squeeze(mriImg(:,:,nslice)),1);
                elseif (length(imgsize) == 4)
                    if acqpInfo.nEvolutionCycles >1,
                        for dceframe=1:acqpInfo.nEvolutionCycles
                            mrirot(:,:,nslice,dceframe)=rot90(squeeze(mriImg(:,:,nslice,dceframe)),1);
                        end % for each frame, DCE
                    end
                    if acqpInfo.nEvolutionCycles == 1 && acqpInfo.nEchoes > 1                        
                        for myEcho=1:acqpInfo.nEchoes
                            mrirot(:,:,myEcho,nslice)=rot90(squeeze(mriImg(:,:,myEcho,nslice)),1);
                        end % for each frame, echo
                    end                    
                end % if 3D
            end % for each slice
            mriImg = mrirot; clear mrirot
        end % if axial and not spectrum
    end % if PV5
    if (acqpInfo.FOV(1) == 0 && acqpInfo.sliceThick == 0) || (acqpInfo.nSlices == 0 && acqpInfo.sliceSep == 0)
        % spectrum (1D) or PRESS
        mriImg=mriImgin;
    end
    
    % if image is Prone but ParaVision thinks Supine, then axial image
    % still OK.  It's mostly a display problem.  CH 02-07-07
    % may display wrong if set(gca,'YDir','Normal')
    % **************
    % April 4, 2007: now checks if image is both HeadFirst and axial, not
    % just axial.
    % Dec 4, 2007: If image is FeetFirst and Axial it is reconstructed and
    % imported into Matlab but needs permute X-Y.  Sagittal needs to be reversed left/right (flipdim, 3)
    % Coronal image is wrong if user states Supine when
    % sample is Prone.  Sagittal and axial don't
    % seem to be affected by supine vs prone.
    % based on "orientation" phantom imaged on 12-01-07
    % allow for anisotropic FOV CH 01-10-08
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    % if user says Head 1st and sample is 1st, then still need rot90
    % clockwise Nov 12, 2008 CH

end  % end if fid <=0
varargout{1} = mriImg;
if nargin == 1 || (nargin >= 2 && isempty(varargin{1}))
    varargout{2} = acqpInfo;
end
% if img3D==1 &&  imgsize(3) < 96 && nargout >=2 %if this is 3D image, then resample it to 128^3
%     [xi,yi,zi] = meshgrid(1:acqpInfo.matrix(1),1:acqpInfo.matrix(2),linspace(1,acqpInfo.matrix(3),128));
%     mriImg128=interp3(mriImg,xi,yi,zi);
%     if nargout == 3
%         varargout{3} = mriImg128;
%     elseif nargin > 2  && nargout == 2
%         varargout{2} = mriImg128;
%     end
% end