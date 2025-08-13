function varargout = t1mapper(imgfn,nobjIn,mythresh,musID,func2fit)
% make t1 or t2 map from Bruker rareVTR_T1 (sat. recovery) or MSME (T2).
% give image filename (2dseq), number of objects in the image, mouse (or exp) ID
% then it will return raw t1 image (series of TRs or TEs),
% image info, T1 map, and errors-map from fit.  T1 bounds fixed at 40 to 3500
% modified Nov 2007 to collate two images, one with low TRs and the other with high TRs.
% have to "normalize" images before collating
% Modified to work with background mask from DCE setup 8-29-08 CH.
% Add option to pick T1sat (saturation recovery), T1inv (inversion recov.), or T2 map (function to fit).  4-8-09 CH.
% example:
% [rws41d3T1Img,rws41d3T1, rws41d3T1map, rws41d3T1mapError] = ...
%     t1mapper('D:\MRI_data\rws41_11-14-08_d3\wei111408.Qm1\7\pdata...
%     \1\rws41d3_rare7Ax.img',[2 3 3],0.15,'rws41d3');
%  Changed from ms to sec CH May 11, 2010

% Need to make a version that takes muscle and tumor masks from multiple
% slices and a single T1 map and make multiple slice T1 map based on avg T1
% for muslce and tumor ROIs.




% !!!!!!!!!!!!!!!  need to fix for T2 (nEchoes vs nEvolutions), fix fit
% limits
wd=cd;
if ~iscell(imgfn), imgfn={imgfn}; end  % assume only one image passed, convert to cell for compat
if isempty(func2fit), func2fit='T1sat'; end % make default function T1 map from sat recov

diary([wd,'\', musID,'T1mapResults.txt']);

% for nImg = 1:size(imgfn,1)
%     eval(['[t1img',nImg,',acqpInfo',nImg,']=read2dseq(imgfn);'])
% end
[t1img1,acqpInfo1]=read2dseq(imgfn{1});
if size(imgfn,1) == 1, colImginfo=acqpInfo1; end
if size(imgfn,1) == 1, T1img=t1img1; clear t1img1; end
if size(imgfn,1) == 2
    [t1img2,acqpInfo2]=read2dseq(imgfn{2}); %[],{'HeadFirst','Supine'}
    [T1img,colImginfo] = collateT1Img(t1img1,acqpInfo1,t1img2,acqpInfo2);
    clear t1img1 t1img2
elseif size(imgfn,1) > 2
    error('only setup to handle two files for now')
end
% % truncate T1 image per Greg's test
% T1img=T1img(:,:,:,1:end-2);
% colImginfo.nEvolutionCycles=colImginfo.nEvolutionCycles-2;
% colImginfo.repTime=colImginfo.repTime(1:end-2);
pst=CalcAxesPos(1,2,[], []);
% setup figure positions
%[left, bottom, width, height]:

% pos1 = round([1, myscrn(4)/2-120, myscrn(3)/3, myscrn(4)/2]);
% pos2 = round([myscrn(3)/3, myscrn(4)/2-120, myscrn(3)/3, myscrn(4)/2]);
% pos3 = round([2*myscrn(3)/3, myscrn(4)/2-120, myscrn(3)/3, myscrn(4)/2]);
% 
% pos4 = round([1, 1, myscrn(3)/3, myscrn(4)/2]);
% pos5 = round([myscrn(3)/3, 1, myscrn(3)/3, myscrn(4)/2]);
% pos6 = round([2*myscrn(3)/3, 1, myscrn(3)/3, myscrn(4)/2]);
% 
% set(53,'position',pos1)
% set(34,'position',pos4)
% 
% set(54,'position',pos2)
% set(35,'position',pos5)
% 
% set(55,'position',pos3)
% set(36,'position',pos6)

set(0,'Units','pixels');
myscrn=get(0,'ScreenSize');
figpos = ones(colImginfo.nSlices*2,4);
figpos(:,3) = round(myscrn(3)/colImginfo.nSlices)*0.95;
figpos(:,4) = round(myscrn(4)/2)*0.95;
figpos(1:colImginfo.nSlices,2) = round(myscrn(4)/2-120); % assume start menu at top
for ns=2:colImginfo.nSlices
    figpos(ns,1) = round(myscrn(3)/colImginfo.nSlices*(ns-1));
    figpos(ns+colImginfo.nSlices,1) = round(myscrn(3)/colImginfo.nSlices*(ns-1));
end

t1map = zeros([colImginfo.matrix(1:2) colImginfo.nSlices]);
t1error = zeros([colImginfo.matrix(1:2) colImginfo.nSlices]);
% process each slice
for sliceN=1:colImginfo.nSlices
    if length(nobjIn) >1
        nobj=nobjIn(sliceN);
    else
        nobj=nobjIn;
    end
    t1img=squeeze(T1img(:,:,sliceN,:));
    % show longest TR image
    maxTRimg=squeeze(t1img(:,:,colImginfo.nEvolutionCycles));
    figure(52+sliceN);
    set(52+sliceN,'position',figpos(sliceN+colImginfo.nSlices,:))
    axes('Position',pst(1,:));imagesc(maxTRimg); axis image
    % tumor displayed correctly if data is new i.e., permute x-y
    set(gca,'YDir','Normal') % PV 4.0 display OK w/ default Matlab, CH 12-05-07
    % set(gca,'YDir','Reverse') will not work see phantom Dec 1 2007
    % make background mask
    t1imgBack=false(colImginfo.matrix(1),colImginfo.matrix(2));
    % use outside_mask if only one object
    %     t1imgBack=outside_mask(maxTRimg,0.075);
    mymax=max(maxTRimg(:));
    %     t1imgBack(maxTRimg >= mymax*0.075)=1;
    if ~exist('bkmskmat','var') % is mask loaded, i.e., not first slice iteration
        backFileYN = questdlg('Load from file?','Load Masks');
    else
        backFileYN = 'yes';
    end
    
    if strcmpi(backFileYN,'no')
        bwI=connected_components(auto_mask_volume(maxTRimg),mythresh,nobj);
        if size(bwI,1) < nobj, error(['Only ', num2str(size(bwI,1)), ' object(s) found']), end
        for nn=1:nobj
            t1imgBack=t1imgBack+bwI{nn}*nn;
        end
    elseif strcmpi(backFileYN,'yes')
        if ~exist('bkmskmat','var') % check if loaded already
            % assumes that background mask is part of muscle ROI file
            [bkmskmat,dirstring]=uigetfile({'*.mat;*.MAT','Muscle Pts (*.mat)';'*.*','All files (*.*)'},'Select Back Mask file');
            bkmskmat = [dirstring, bkmskmat]; clear dirstring
            eval(['bkmskVarN=whos(''-file'', ''',bkmskmat,''');'])
            load(bkmskmat)
            for z=1:size(bkmskVarN,1)
                % assume background mask is logical and msl pts is cell
                if strcmp(bkmskVarN(z).class,'cell')
                    eval(['mslROIdata=' bkmskVarN(z).name,';']);
                elseif eval(['islogical(',bkmskVarN(z).name,')'])
                    eval(['legmask=' bkmskVarN(z).name,';']);
                end
                eval(['clear ', bkmskVarN(z).name])
            end
        end
        t1imgBack=squeeze(legmask(:,:,sliceN));
        % need to combine fiducials and leg
        if nobj > 1 % leg plus fids
            NoLeg=maxTRimg;
            NoLeg(t1imgBack)=0; % remove leg mask in case fid is too close
            NoLeg=NoLeg/max(NoLeg(:));
            NoLeg(NoLeg < mythresh)=0;
            bwI=connected_components(NoLeg,mythresh,nobj-1);  %nobj - leg
            if size(bwI,1) < nobj-1, error(['Only ', num2str(size(bwI,1)), ' object(s) found']), end
            for nn=1:nobj-1
                t1imgBack=t1imgBack+bwI{nn}*(nn+1);
            end
            bwI(nobj)={squeeze(legmask(:,:,sliceN))}; % add leg back
            bwI=bwI([nobj 1:nobj-1]); % sort labels
            %         else  % leg only
            %             % add blank mask to leg
            %             bwI={zeros(size(t1imgBack))};
        else
            bwI(nobj)={squeeze(legmask(:,:,sliceN))};
        end
    end % if user wants to load back mask file
    
    clear maxTRimg nn
    
    % figure; imagesc(t1imgBack); axis image
    drawnow
    % set options for lsqcurvefit
    optionsdef = optimset('lsqcurvefit');
    optionsnew = optimset('Diagnostics', 'off','Display', 'off','MaxFunEvals',100,'MaxIter',...
        100,'LargeScale','on','TolX',1e-6,'TolFun',1e-9);
    options = optimset(optionsdef,optionsnew);
    
    switch lower(func2fit)
    case 't1sat'
        % saturation recovery equation to fit
        % change limits from ms to sec CH 05-11-10
        % Bruker's version has a 'bias' variable but no 'beta' in front of
        % the exponential
%         Func2Fit = @(a,t) a(3)+abs(a(1)*(1-exp(-t/a(2))));        
%         pars0 = [mymax 1.99 mymax/2];
%         lb = [0 0.040 -mymax*0.75];
%         ub = [mymax*1.5 3.5 mymax*0.75];
        % Standard  sat recov function with 'beta' variable in front of the
        % exponential.  It should be close to 1.  CH 05-13-10
        Func2Fit = @(a,t) a(1)*(1-a(3)*exp(-t/a(2)));
        pars0 = [mymax 1.99 1];
        lb = [0 0.040 0.5];
        ub = [mymax*1.5 3.5 1.2];
        
    case 't1inv'
        % inversion recovery equation to fit
        % change limits from ms to sec CH 05-11-10
        Func2Fit = @(a,t) a(1)*(1-2*exp(-t/a(2)));
        pars0 = [mymax 1.99];
        lb = [0 0.040];
        ub = [mymax*1.5 3.5];
        
    case 't2'
        % T2 relaxation: y=A+C*exp(-t/T2)
        Func2Fit = @(a,t) a(3)+abs(a(1)*(exp(-t/a(2))));
        % need to make limits        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        pars0 = [mymax 1990 mymax/2];
        lb = [0 40 -mymax*0.75];
        ub = [mymax*1.5 3500 mymax*0.75];
    end

    % label image
    %     [bwI,nobj] = bwlabel(t1imgBack,4);
    % fit each object
    
    t1Avg = zeros(colImginfo.matrix(1),colImginfo.matrix(1));
    SmaxAvg = zeros(colImginfo.matrix(1),colImginfo.matrix(1));
    biasAvg = zeros(colImginfo.matrix(1),colImginfo.matrix(1));
    
    for n=1:nobj
        [r,c] = find(bwI{n});
        
        if ~isempty(r)
            tempS=zeros(length(r),colImginfo.nEvolutionCycles);
            for x=1:length(r)
                tempS(x,:)=t1img(r(x),c(x),:);
            end
            if numel(tempS) > colImginfo.nEvolutionCycles
                meanS=mean(tempS,1);
            else
                meanS=tempS;
            end
            figure(44+sliceN);
            % change TR from ms to sec CH 05-11-10
            [pars, resnorm,residual,exitflag,output] = lsqcurvefit(Func2Fit,pars0,...
                colImginfo.repTime/1e3,meanS,lb,ub,options);
            
            %             disp(['LSQ Exit flag: ', num2str(exitflag)])
            %             disp(['LSQ ResNorm: ', num2str(resnorm)])
            %             disp(['LSQ Iterations: ', num2str(output.iterations)])
            %             disp(['LSQ Func. Calls: ', num2str(output.funcCount)])
            %             disp(output.message)
            
            plot(colImginfo.repTime/1e3,meanS,'ro')
            hold on
            plot(colImginfo.repTime/1e3,Func2Fit(pars,(colImginfo.repTime')/1e3),'-b')
            myTitle=sprintf('Smax: %6.0f, T1: %6.2f, bias: %6.2f, eFlag: %d',pars,exitflag);
            title(myTitle)
            for x=1:length(r)
                SmaxAvg(r(x),c(x)) = pars(1);
                t1Avg(r(x),c(x)) = pars(2);
                biasAvg(r(x),c(x)) = pars(3);
            end
            clear pars resnorm residual exitflag output
            drawnow
            pause(0.3)
        end
    end
    % fit each pixel
    optionsdef = optimset('lsqcurvefit');
    optionsnew = optimset('Diagnostics', 'off','Display', 'off','MaxFunEvals',60,'MaxIter',...
        20,'LargeScale','on','TolX',1e-6,'TolFun',1e-9);
    options = optimset(optionsdef,optionsnew);
    % use mean of each object as initial values for pixel based fit
    t1 = zeros(colImginfo.matrix(1),colImginfo.matrix(1));
    t1errorslice = zeros(colImginfo.matrix(1),colImginfo.matrix(1));
    [r,c] = find(t1imgBack);
    tic
    figure(15+sliceN);
    testout=round(rand(1,floor(length(r)*0.01))*length(r));
    for x=1:length(r)
        pars0 = [SmaxAvg(r(x),c(x)) t1Avg(r(x),c(x)) biasAvg(r(x),c(x))];
        % change TR from ms to sec CH 05-11-10
        [pars, resnorm,residual,exitflag,output] = lsqcurvefit(Func2Fit,pars0,...
            (colImginfo.repTime')/1e3,squeeze(t1img(r(x),c(x),:)),lb,ub,options);
        if ismember(x,testout)
            clf(15+sliceN)
            %             disp(['LSQ Iterations: ', num2str(output.iterations)])
            %             disp(['LSQ Func. Calls: ', num2str(output.funcCount)])
            %             disp(output.message)
            plot((colImginfo.repTime')/1e3,squeeze(t1img(r(x),c(x),:)),'ro')
            hold on
            plot((colImginfo.repTime')/1e3,Func2Fit(pars,(colImginfo.repTime')/1e3),'-b')
            myTitle=sprintf('Smax: %6.0f, T1: %6.2f, bias: %6.2f, eFlag: %d',pars,exitflag);
            title(myTitle)
            pause(0.4)
            drawnow
        end
        t1(r(x),c(x)) = pars(2); clear pars
        t1errorslice(r(x),c(x)) = resnorm;
    end
    toc
    close(15+sliceN)
    
    figure(52+sliceN);
    
    axes('Position',pst(2,:)); imagesc(t1); colorbar('location','SouthOutside'); axis image; axis off;set(gca,'YDir','Normal')
    colormap(jet)
    % display mean for each object
    for n=1:nobj
        fprintf(1,'Mean for object %3.0f is %6.2f [sec] \n',n,mean(t1(bwI{n})))
    end
    fprintf(1,'Mean for muscle slice-%1.0f is %6.2f [sec] \n',sliceN,mean(t1(mslROIdata{3,sliceN})))
    figure(33+sliceN);imagesc(t1imgBack);colorbar; colormap(jet(nobj+1)); axis image; axis off;set(gca,'YDir','Normal')
    set(33+sliceN,'position',figpos(sliceN,:))
    t1map(:,:,sliceN)=t1; clear t1
    t1error(:,:,sliceN)=t1errorslice; clear t1errorslice
    eval(['print -f',num2str(33+sliceN),' -dpng -r150 ', musID, 'S',num2str(sliceN),'T1bwGuide.png'])
    eval(['print -f',num2str(44+sliceN),' -dpng -r150 ', musID, 'S',num2str(sliceN),'T1Fits.png'])
    eval(['print -f',num2str(52+sliceN),' -dpng -r150 ', musID, 'S',num2str(sliceN),'T1map.png'])
end % for each slice
varargout{1} = T1img;
varargout{2} = colImginfo;
varargout{3} = t1map;
if nargout == 4
    varargout{4} = t1error;
end
disp(['save ',musID,'_T1map'])
cd(wd)  %return to working dir.
diary off

function [colImg,colImginfo] = collateT1Img(img1,info1,img2,info2)
% collateT1Img is based on collateImg. it will collate two T1 inv recov images that may have overlapping TRs.
% They need to be windows/scaled the same before collating.
% send "fast" image first
% C Haney November 2007

% find duplicates/overlap
[matchingNums, idximg1in2, idximg2in1]=intersect(info1.repTime,info2.repTime);
% remove overlap
if ~isempty(matchingNums)
    img1(:,:,:,idximg1in2)=[];
    info1.repTime(idximg1in2)=[];
end
% collate images
colImg=cat(4,img1,img2);
colImginfo=info2;
colImginfo.repTime=[info1.repTime info2.repTime];

[TRsorted,idx]=sort(colImginfo.repTime);
colImg=colImg(:,:,:,idx);
colImginfo.repTime=TRsorted;

% remove filename, now irrelevant
colImginfo=rmfield(colImginfo,'filename');
% fix number of TRs
colImginfo.nEvolutionCycles=size(colImg,4);