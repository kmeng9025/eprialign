%% Read in projections from all images
mypath = 'G:\Documents\Projects\DynamicEPRI\AnimalExperiments\131206_airmouse\';
for ii = 1:15
    FBP = [];
    probe_info = [];
    FBP.imtype    = 14;
    FBP.CoordPole = 'Z';
    FBP.nPolar    = 36;
    FBP.nAz       = 36;
    FBP.MaxGradient = 0.75;
    FBP.angle_sampling = 'UNIFORM_SPATIAL_FLIP';
    FBP.baseline = 'every_n';
    FBP.bl_n = 4;
    opt.FBP = FBP;
    opt.Sequence = 'ESEInvRec';
%     EXTRACT PROJECTIONS AND BASELINES
    if ii < 10
        [mat,mat_bl,raw_info] = ...
            epr_ReadPulseImageFile(strcat([mypath,'data\MSPS_mouse_T1_36x36_e4_175shots_reflectionresonator_00',...
            num2str(ii),'.d01']),opt);
        strcat([mypath,'data\MSPS_mouse_T1_36x36_e4_175shots_reflectionresonator_00',...
            num2str(ii),'.d01'])
    else
        [mat,mat_bl,raw_info] = ...
            epr_ReadPulseImageFile(strcat([mypath,'data\MSPS_mouse_T1_36x36_e4_175shots_reflectionresonator_0',...
            num2str(ii),'.d01']),opt);
        strcat([mypath,'data\MSPS_mouse_T1_36x36_e4_175shots_reflectionresonator_0',...
            num2str(ii),'.d01'])
    end
%     SUBTRACT BASELINES FROM PROJECTIONS
    proj{ii} = mat;
    proj_bl{ii} = mat_bl;
    proj_raw_info{ii} = raw_info;
    projections{ii} = proj{ii}-proj_bl{ii};
end
clear proj
clear proj_bl
clear mat
clear mat_bl
save(strcat(mypath,'full_unfiltered_projections.mat'),'projections','proj_raw_info')

%% PCA filter full sets of projections
mypath = 'G:\Documents\Projects\DynamicEPRI\AnimalExperiments\131206_airmouse\';
load(strcat(mypath,'full_unfiltered_projections.mat'))
X = zeros(size(projections{1},1)*size(projections{1},2)*...
    size(projections{1},3),length(projections));
for ii = 1:length(projections)
    X(:,ii) = reshape(projections{ii},size(projections{ii},1)*...
        size(projections{ii},2)*size(projections{ii},3),1);
end
% CHOOSE THE NUMBER OF PRINCIPAL COMPONENTS TO USE IN FILTERING OF
% PROJECTIONS
for PCs = 2:6
    [PCAapprox, eigvec, eigval, score, loading] = GeneralizedPCA(X,1:PCs);
    for ii = 1:length(projections)
        PCAfiltered_projections{ii} = reshape(PCAapprox(:,ii),...
            size(projections{1},1),size(projections{1},2),...
            size(projections{1},3));
    end
    save(strcat(mypath,'full_PCAfiltered_projections_',num2str(max(PCs)),'PCs.mat'),'PCAfiltered_projections','proj_raw_info')
end
%% Reconstruct long image from average of all images
mypath = 'G:\Documents\Projects\DynamicEPRI\AnimalExperiments\131206_airmouse\';
load(strcat(mypath,'full_unfiltered_projections.mat'))
avg_projections = zeros(size(projections{1}));
for ii = 1:length(projections)
    avg_projections = avg_projections + projections{ii};
end
avg_projections = avg_projections/length(projections);

% DEFINE NECESSARY PARAMETERS
FBP = [];
FBP.imtype    = 14;
FBP.CoordPole = 'Z';
FBP.nPolar    = 36;
FBP.nAz       = 36;
FBP.MaxGradient = 0.75;
FBP.angle_sampling = 'UNIFORM_SPATIAL_FLIP';
FBP.baseline = 'every_n';
FBP.bl_n = 4;

mypath2 = 'Z:\Matlab_BE\time_domain\';
cfg = epr_LoadScenario([mypath2, 'PulseRecon.scn'],...
    [mypath2, 'Pulse T1 inversion recovery.par']);
% cfg.td.baseline_algorithm   = 'none';
% cfg.td.use_echos = [];
cfg.fbp.MaxGradient = FBP.MaxGradient;
cfg.fft.FOV = cfg.rec.Size * 2 * FBP.MaxGradient;
% cfg.fft.profile_correction = 'none';
% cfg.td.dead_time = 0;
% cfg.td.acq_window_start = -2000;
cfg.prc.save_data = 'no';
cfg.fbp.nPolar = 36;
cfg.fbp.nAz = 36;
% cfg.rec.Size = cfg.rec.Size/sqrt(2);
cfg.rec.CodeFlag = 'SINGLE';
cfg.rec.Intrp = 'none';
% cfg.rec.FilterCutOff = 1;
%         pars.fit.fit_mask = 'external_file';
%         pars.fit.fit_mask_file = 'G:\DOCUMENTS\Projects\Gradients\UniformAcquisition\Compare_short_vs_intermediate\36x36_modSL\InterpAndFilterOff_setAMP_4subvolumes_12x12_18x18_36x36_72x72\noiselessfull72_Mask.mat';

load('G:\Documents\Projects\MSPS\UniformAcquisition\GradientSchemes\36x36_MinU_idx.mat')
tempinfo = proj_raw_info{1};
% REARRANGE X,Y,Z COMPONENTS TO MATCH MSPS ACQUIRED PROJECTIONS
tempinfo.GradX = tempinfo.GradX(idx);
tempinfo.GradY = tempinfo.GradY(idx);
tempinfo.GradZ = tempinfo.GradZ(idx);

% RECONSTRUCT T1 IMAGES USING ese_fbp_InvRe
full_unfiltered_avg_allimages = ese_fbp_InvRec(struct('mat', avg_projections, 'raw_info', tempinfo), '','',cfg);
save(strcat(mypath,'full_unfiltered_avg_allimages.mat'),'full_unfiltered_avg_allimages')
Mask = full_unfiltered_avg_allimages.Mask;
save(strcat(mypath,'full_unfiltered_avg_allimages_Mask.mat'),'Mask')

cfg.fit.fit_mask = 'external_file';
cfg.fit.fit_mask_file = strcat(mypath,'full_unfiltered_avg_allimages_Mask.mat');
for PCs = 2:6
    load(strcat(mypath,'full_PCAfiltered_projections_',num2str(PCs),'PCs.mat'))
    avg_PCAfiltered_projections = zeros(size(PCAfiltered_projections{1}));
    for ii = 1:length(PCAfiltered_projections)
        avg_PCAfiltered_projections = avg_PCAfiltered_projections + PCAfiltered_projections{ii};
    end
    avg_PCAfiltered_projections = avg_PCAfiltered_projections/length(PCAfiltered_projections);
    full_PCAfiltered_avg_allimages = ese_fbp_InvRec(struct('mat', avg_PCAfiltered_projections, 'raw_info', tempinfo), '','',cfg);
    save(strcat(mypath,'full_PCAfiltered_avg_allimages_',num2str(max(PCs)),'PCs.mat'),'full_PCAfiltered_avg_allimages')
end

%% Reconstruct full images
mypath = 'G:\Documents\Projects\DynamicEPRI\AnimalExperiments\131206_airmouse\';
FBP = [];
FBP.imtype    = 14;
FBP.CoordPole = 'Z';
FBP.nPolar    = 36;
FBP.nAz       = 36;
FBP.MaxGradient = 0.75;
FBP.angle_sampling = 'UNIFORM_SPATIAL_FLIP';
FBP.baseline = 'every_n';
FBP.bl_n = 4;

mypath2 = 'Z:\Matlab_BE\time_domain\';
cfg = epr_LoadScenario([mypath2, 'PulseRecon.scn'],...
    [mypath2, 'Pulse T1 inversion recovery.par']);
% cfg.td.baseline_algorithm   = 'none';
% cfg.td.use_echos = [];
cfg.fbp.MaxGradient = FBP.MaxGradient;
cfg.fft.FOV = cfg.rec.Size * 2 * FBP.MaxGradient;
% cfg.fft.profile_correction = 'none';
% cfg.td.dead_time = 0;
% cfg.td.acq_window_start = -2000;
cfg.prc.save_data = 'no';
cfg.fbp.nPolar = 36;
cfg.fbp.nAz = 36;
% cfg.rec.Size = cfg.rec.Size/sqrt(2);
cfg.rec.CodeFlag = 'SINGLE';
cfg.rec.Intrp = 'none';
% cfg.rec.FilterCutOff = 1;
cfg.fit.fit_mask = 'external_file';
cfg.fit.fit_mask_file = strcat(mypath,'full_unfiltered_avg_allimages_Mask.mat');

load('G:\Documents\Projects\MSPS\UniformAcquisition\GradientSchemes\36x36_MinU_idx.mat')
tempinfo = proj_raw_info{1};
tempinfo.GradX = tempinfo.GradX(idx);
tempinfo.GradY = tempinfo.GradY(idx);
tempinfo.GradZ = tempinfo.GradZ(idx);
% RECONSTRUCT INDIVIDUAL FULL IMAGES
for ii = 1:length(projections)
    temp = ese_fbp_InvRec(struct('mat', projections{ii}, 'raw_info', tempinfo), '','',cfg);
    if ii == 1
        full_unfiltered.raw_info = temp.raw_info;
        full_unfiltered.rec_info = temp.rec_info;
        full_unfiltered.Size = temp.Size;
        full_unfiltered.Amp = temp.Amp;
        full_unfiltered.T1 = temp.T1;
        full_unfiltered.Mask = temp.Mask;
        full_unfiltered.Error = temp.Error;
        full_unfiltered.Error_T1 = temp.Error_T1;
        full_unfiltered.pO2 = temp.pO2;
    else
        full_unfiltered.Amp(:,:,:,ii) = temp.Amp;
        full_unfiltered.T1(:,:,:,ii) = temp.T1;
        full_unfiltered.Error(:,:,:,ii) = temp.Error;
        full_unfiltered.Error_T1(:,:,:,ii) = temp.Error_T1;
        full_unfiltered.pO2(:,:,:,ii) = temp.pO2;
    end
    clear temp
end
save(strcat(mypath,'full_unfiltered.mat'),'full_unfiltered')

for PCs = 2:6
    load(strcat(mypath,'full_PCAfiltered_projections_',num2str(PCs),'PCS.mat'))
    load('G:\Documents\Projects\MSPS\UniformAcquisition\GradientSchemes\36x36_MinU_idx.mat')
    tempinfo = proj_raw_info{1};
    tempinfo.GradX = tempinfo.GradX(idx);
    tempinfo.GradY = tempinfo.GradY(idx);
    tempinfo.GradZ = tempinfo.GradZ(idx);
    for ii = 1:length(PCAfiltered_projections)
        temp = ese_fbp_InvRec(struct('mat', PCAfiltered_projections{ii}, 'raw_info', tempinfo), '','',cfg);
        if ii == 1
            full_PCAfiltered.raw_info = temp.raw_info;
            full_PCAfiltered.rec_info = temp.rec_info;
            full_PCAfiltered.Size = temp.Size;
            full_PCAfiltered.Amp = temp.Amp;
            full_PCAfiltered.T1 = temp.T1;
            full_PCAfiltered.Mask = temp.Mask;
            full_PCAfiltered.Error = temp.Error;
            full_PCAfiltered.Error_T1 = temp.Error_T1;
            full_PCAfiltered.pO2 = temp.pO2;
        else
            full_PCAfiltered.Amp(:,:,:,ii) = temp.Amp;
            full_PCAfiltered.T1(:,:,:,ii) = temp.T1;
            full_PCAfiltered.Error(:,:,:,ii) = temp.Error;
            full_PCAfiltered.Error_T1(:,:,:,ii) = temp.Error_T1;
            full_PCAfiltered.pO2(:,:,:,ii) = temp.pO2;
        end
        clear temp
    end
    save(strcat(mypath,'full_PCAfiltered_',num2str(PCs),'PCs.mat'),'full_PCAfiltered')
end

%% PCA filter grouped divisions of projections
clear X
mypath = 'G:\Documents\Projects\DynamicEPRI\AnimalExperiments\131206_airmouse\';
load(strcat(mypath,'full_unfiltered_projections.mat'))
for PCs = 2:6
    for numdiv = [2 3 4 6 9 12 18 36]
        for jj = 1:numdiv
            for ii = 1:length(projections)
                X{jj}(:,ii) = reshape(projections{ii}(:,:,(jj-1)*(length(proj_raw_info{ii}.GradX)/numdiv)+1:...
                    jj*(length(proj_raw_info{ii}.GradX)/numdiv)),...
                    size(projections{ii},1)*size(projections{ii},2)*size(projections{ii},3)/numdiv,1);
            end
            [PCAapprox{jj}, eigvec{jj}, eigval{jj}, score{jj}, loading{jj}] = GeneralizedPCA(X{jj},1:PCs);
            clear temp temp1 temp2 X
        end
        for jj = 1:numdiv
            for ii = 1:length(projections)
                temp1 = PCAapprox{jj}(:,ii);
                temp{jj}{ii} = ...
                    reshape(temp1,size(projections{ii}(:,:,(jj-1)*...
                    (length(proj_raw_info{ii}.GradX)/numdiv)+1:jj*...
                    (length(proj_raw_info{ii}.GradX)/numdiv))));
            end
        end
        for ii = 1:length(projections)
            for jj = 1:numdiv
                temp2{ii}(:,:,(jj-1)*(length(proj_raw_info{ii}.GradX)/numdiv)+1:jj*(length(proj_raw_info{ii}.GradX)/numdiv)) = temp{jj}{ii};
            end
        end
        PCAapprox = temp2;
        clear temp temp1 temp2
        
        save(strcat(mypath,num2str(numdiv),'div_groupPCAfiltered_proj_',num2str(PCs),'PCs.mat'), 'PCAapprox', 'proj_raw_info','eigvec','eigval')
    end
end

%% Reconstruct divisions
mypath = 'G:\Documents\Projects\DynamicEPRI\AnimalExperiments\131206_airmouse\';
load(strcat(mypath,'full_unfiltered_projections.mat'))
for numdiv = [2 3 4 6 9]
    FBP = [];
    FBP.imtype    = 14;
    FBP.CoordPole = 'Z';
    FBP.nPolar    = 36;
    FBP.nAz       = 36;
    FBP.MaxGradient = 0.75;
    FBP.angle_sampling = 'UNIFORM_SPATIAL_FLIP';
    FBP.baseline = 'every_n';
    FBP.bl_n = 4;
    
    mypath2 = 'Z:\Matlab_BE\time_domain\';
    cfg = epr_LoadScenario([mypath2, 'PulseRecon.scn'],...
        [mypath2, 'Pulse T1 inversion recovery.par']);
    % cfg.td.baseline_algorithm   = 'none';
    % cfg.td.use_echos = [];
    cfg.fbp.MaxGradient = FBP.MaxGradient;
    cfg.fft.FOV = cfg.rec.Size * 2 * FBP.MaxGradient;
    % cfg.fft.profile_correction = 'none';
    % cfg.td.dead_time = 0;
    % cfg.td.acq_window_start = -2000;
    cfg.prc.save_data = 'no';
    cfg.fbp.nPolar = 36;
    cfg.fbp.nAz = 36;
    % cfg.rec.Size = cfg.rec.Size/sqrt(2);
    cfg.rec.CodeFlag = 'SINGLE';
    cfg.rec.Intrp = 'none';
    % cfg.rec.FilterCutOff = 1;
    cfg.fit.fit_mask = 'external_file';
    cfg.fit.fit_mask_file = strcat(mypath,'full_unfiltered_avg_allimages_Mask.mat');
    
%     load(strcat('G:\Documents\Projects\MSPS\UniformAcquisition\GradientSchemes\MSPS_828_approxvoronoi\MSPS_828proj_ApproxVor_',num2str(numdiv),'divs.mat'))
    load('G:\Documents\Projects\MSPS\UniformAcquisition\GradientSchemes\36x36_MinU_idx.mat')
    
%     SUBDIVIDE AND RECONSTRUCT IMAGES FROM SUBSETS OF PROJECTIONS
    count = 0;
    for ii = 1:length(projections)
        for jj = 1:numdiv
            count = count + 1;
            tempinfo = proj_raw_info{ii};
            tempmat = projections{ii}(:,:,(jj-1)*size(projections{ii},3)/numdiv+1:...
                jj*size(projections{ii},3)/numdiv);
            tempinfo.GradX = ...
                tempinfo.GradX(idx((jj-1)*size(projections{ii},3)/numdiv+1:...
                jj*size(projections{ii},3)/numdiv));
            tempinfo.GradY = ...
                tempinfo.GradY(idx((jj-1)*size(projections{ii},3)/numdiv+1:...
                jj*size(projections{ii},3)/numdiv));
            tempinfo.GradZ = ...
                tempinfo.GradZ(idx((jj-1)*size(projections{ii},3)/numdiv+1:...
                jj*size(projections{ii},3)/numdiv));
            
%             CALCULATE WEIGHTS FOR PROJECTIONS BASED ON VORONOI WEIGHTS
            areas = vor_area_3d(tempinfo.GradX,tempinfo.GradY,tempinfo.GradZ);
            areas = areas/sum(areas);
            cfg.fbp.Wt = areas;
            
            temp = ese_fbp_InvRec(struct('mat', tempmat, 'raw_info', tempinfo), '','',cfg);
            if count == 1
                div_unfiltered.raw_info = temp.raw_info;
                div_unfiltered.rec_info = temp.rec_info;
                div_unfiltered.Size = temp.Size;
                div_unfiltered.Amp = temp.Amp;
                div_unfiltered.T1 = temp.T1;
                div_unfiltered.Mask = temp.Mask;
                div_unfiltered.Error = temp.Error;
                div_unfiltered.Error_T1 = temp.Error_T1;
                div_unfiltered.pO2 = temp.pO2;
            else
                div_unfiltered.Amp(:,:,:,count) = temp.Amp;
                div_unfiltered.T1(:,:,:,count) = temp.T1;
                div_unfiltered.Error(:,:,:,count) = temp.Error;
                div_unfiltered.Error_T1(:,:,:,count) = temp.Error_T1;
                div_unfiltered.pO2(:,:,:,count) = temp.pO2;
            end
            clear temp tempmat tempinfo
        end
    end
    save(strcat(mypath,num2str(numdiv),'div_unfiltered.mat'),'div_unfiltered')
    
    for PCs = 2:6;
        load(strcat(mypath,num2str(numdiv),'div_groupPCAfiltered_proj_',num2str(PCs),'PCs.mat'))
        count = 0;
        for ii = 1:length(PCAapprox)
            for jj = 1:numdiv
                count = count + 1;
                tempinfo = proj_raw_info{ii};
                tempmat = PCAapprox{ii}(:,:,(jj-1)*size(projections{ii},3)/numdiv+1:...
                    jj*size(projections{ii},3)/numdiv);
                tempinfo.GradX = ...
                    tempinfo.GradX(idx((jj-1)*size(projections{ii},3)/numdiv+1:...
                    jj*size(projections{ii},3)/numdiv));
                tempinfo.GradY = ...
                    tempinfo.GradY(idx((jj-1)*size(projections{ii},3)/numdiv+1:...
                    jj*size(projections{ii},3)/numdiv));
                tempinfo.GradZ = ...
                    tempinfo.GradZ(idx((jj-1)*size(projections{ii},3)/numdiv+1:...
                    jj*size(projections{ii},3)/numdiv));
                
                areas = vor_area_3d(tempinfo.GradX,tempinfo.GradY,tempinfo.GradZ);
                areas = areas/sum(areas);
                cfg.fbp.Wt = areas;
                
                temp = ese_fbp_InvRec(struct('mat', tempmat, 'raw_info', tempinfo), '','',cfg);
                if count == 1
                    div_PCAfiltered.raw_info = temp.raw_info;
                    div_PCAfiltered.rec_info = temp.rec_info;
                    div_PCAfiltered.Size = temp.Size;
                    div_PCAfiltered.Amp = temp.Amp;
                    div_PCAfiltered.T1 = temp.T1;
                    div_PCAfiltered.Mask = temp.Mask;
                    div_PCAfiltered.Error = temp.Error;
                    div_PCAfiltered.Error_T1 = temp.Error_T1;
                    div_PCAfiltered.pO2 = temp.pO2;
                else
                    div_PCAfiltered.Amp(:,:,:,count) = temp.Amp;
                    div_PCAfiltered.T1(:,:,:,count) = temp.T1;
                    div_PCAfiltered.Error(:,:,:,count) = temp.Error;
                    div_PCAfiltered.Error_T1(:,:,:,count) = temp.Error_T1;
                    div_PCAfiltered.pO2(:,:,:,count) = temp.pO2;
                end
                clear temp tempmat tempinfo
            end
        end
        save(strcat(mypath,num2str(numdiv),'div_groupPCAfiltered_',num2str(PCs),'PCs.mat'),'div_PCAfiltered')
    end
end

% %% Create time axis
% mypath = 'G:\Documents\Projects\DynamicEPRI\AnimalExperiments\130916_airmouse\';
% fullimagetimes = [700 700 700 700 790 700 760 825 700 740 700 740 700 700 700];
% interpauses = [0 0 0 0 0 240 0 0 0 0 0 0 0 0 0]; %pauses before images
% for numdiv = [1 2 3 4 6 9 12 18 36]
%     temppauses = zeros(1,length(fullimagetimes)*numdiv);
%     for ii = 1:length(fullimagetimes)
%         temppauses(ii*numdiv-(numdiv-1)) = interpauses(ii);
%         tempimagetimes(ii*numdiv-(numdiv-1):ii*numdiv) = fullimagetimes(ii)/numdiv;
%     end
%     timeaxis = zeros(size(tempimagetimes));
%     for ii = 1:length(tempimagetimes)
%         timeaxis(ii) = sum(tempimagetimes(1:ii)) + sum(temppauses(1:ii));
%     end
%     clear temppauses tempimagetimes
%     save(strcat(mypath,'timeaxis_',num2str(numdiv),'div.mat'),'timeaxis')
% end
% 
% 
% %% Analysis: active voxels: (31,39,32) inactive voxels: (31,41,34) (33,39,33) (28,38,34)
% for numdiv = [2 3 4 6]
%     load('V:\data\Imagnet_data_13\09\130916\TumorMask130917.mat')
%     tumormask = Mask;
%     load(strcat('G:\Documents\Projects\DynamicEPRI\AnimalExperiments\130916_airmouse\',num2str(numdiv),'div_unfiltered.mat'))
%     unfilt.pO2 = div_unfiltered.pO2;
%     unfilt.mask = div_unfiltered.Mask;
%     load(strcat('G:\Documents\Projects\DynamicEPRI\AnimalExperiments\130916_airmouse\',num2str(numdiv),'div_groupPCAfiltered_6PCs.mat'))
%     PCAfilt.pO2 = div_PCAfiltered.pO2;
%     PCAfilt.mask = div_PCAfiltered.Mask;
%     load(strcat('G:\Documents\Projects\DynamicEPRI\AnimalExperiments\130916_airmouse\timeaxis_',num2str(numdiv),'div.mat'))
%     
%     %average tumor voxels to find dominant pattern:
%     avg_unfilt = zeros(size(unfilt.pO2,4),1);
%     avg_PCAfilt = zeros(size(unfilt.pO2,4),1);
%     count = 0;
%     for ii = 1:size(unfilt.pO2,1)
%         for jj = 1:size(unfilt.pO2,2)
%             for kk = 1:size(unfilt.pO2,3)
%                 if unfilt.mask(ii,jj,kk) == 1 && PCAfilt.mask(ii,jj,kk) == 1 && Mask(ii,jj,kk) == 1
%                     count = count + 1;
%                     avg_unfilt = avg_unfilt + squeeze(unfilt.pO2(ii,jj,kk,:));
%                     avg_PCAfilt = avg_PCAfilt + squeeze(PCAfilt.pO2(ii,jj,kk,:));
%                 end
%             end
%         end
%     end
%     avg_unfilt = avg_unfilt/count;
%     avg_PCAfilt = avg_PCAfilt/count;
%     figure
%     plot(timeaxis,avg_unfilt,'-o',timeaxis,avg_PCAfilt,'-o')
%     unfilt.Excitation = avg_unfilt;
%     PCAfilt.Excitation = avg_unfilt;
%     
%     %calculate standard deviation in each voxel
%     count = 0;
%     for ii = 1:64
%         for jj = 1:64
%             for kk = 1:64
%                 unfilt_temporalstd(ii,jj,kk) = std(squeeze(unfilt.pO2(ii,jj,kk,:)));
%                 PCAfilt_temporalstd(ii,jj,kk) = std(squeeze(PCAfilt.pO2(ii,jj,kk,:)));
%             end
%         end
%     end
%     
%     figure
%     x = 31; y = 39; z = 32;
%     count = 0;
%     clear temp1 temp2
%     for ii = -1:1
%         for jj = -1:1
%             for kk = -1:1
%                 count = count + 1;
%                 temp1(count,:) = squeeze(unfilt.pO2(x+ii,y+jj,z+kk,:));
%                 temp2(count,:) = squeeze(PCAfilt.pO2(x+ii,y+jj,z+kk,:));
%             end
%         end
%     end
%     unfiltmean1 = mean(temp1,1);
%     unfiltstd1 = std(temp1,1);
%     PCAfiltmean1 = mean(temp2,1);
%     PCAfiltstd1 = std(temp2,1);
%     x = 33; y = 39; z = 33;
%     count = 0;
%     for ii = -1:1
%         for jj = -1:1
%             for kk = -1:1
%                 count = count + 1;
%                 temp1(count,:) = squeeze(unfilt.pO2(x+ii,y+jj,z+kk,:));
%                 temp2(count,:) = squeeze(PCAfilt.pO2(x+ii,y+jj,z+kk,:));
%             end
%         end
%     end
%     unfiltmean2 = mean(temp1,1);
%     unfiltstd2 = std(temp1,1);
%     PCAfiltmean2 = mean(temp2,1);
%     PCAfiltstd2 = std(temp2,1);
% %     plot(timeaxis,PCAfiltmean1,'-o',timeaxis,PCAfiltmean2,'-o',timeaxis,unfiltmean1,'-o',timeaxis,unfiltmean2,'-o')
%     plot(timeaxis,PCAfiltmean1,'-o',timeaxis,unfiltmean1,'-o')
% end
% 
% %% Compare time traces for different number divisions:
% %unfiltered
% for numdiv = [1 2 3 4 6 9 12]
%     load(strcat('G:\Documents\Projects\DynamicEPRI\AnimalExperiments\130916_airmouse\timeaxis_',num2str(numdiv),'div.mat'))
%     if numdiv == 1
%         load('G:\Documents\Projects\DynamicEPRI\AnimalExperiments\130916_airmouse\full_unfiltered.mat')
%         Image = full_unfiltered.pO2;
%     elseif numdiv == 12
%         load('G:\Documents\Projects\DynamicEPRI\AnimalExperiments\130916_airmouse\12div_unfiltered(pO2).mat')
%         Image = div_unfiltered_pO2;
%     else
%         load(strcat('G:\Documents\Projects\DynamicEPRI\AnimalExperiments\130916_airmouse\',num2str(numdiv),'div_unfiltered.mat'))
%         Image = div_unfiltered.pO2;
%     end
%     
%     x = 31; y = 39; z = 32;
%     timetrace = zeros(27,size(Image,4));
%     count = 0;
%     for ii = -1:1
%         for jj = -1:1
%             for kk = -1:1
%                 count = count + 1;
%                 timetrace(count,:) = squeeze(Image(x+ii,y+jj,z+kk,:));
%             end
%         end
%     end
%     meantimetrace{numdiv} = mean(timetrace,1);
%     temptaxis{numdiv} = timeaxis;
% end
% figure
% plot(temptaxis{1}/60,meantimetrace{1},...
%     temptaxis{2}/60,meantimetrace{2},...
%     temptaxis{3}/60,meantimetrace{3},...
%     temptaxis{4}/60,meantimetrace{4},...
%     temptaxis{6}/60,meantimetrace{6},...
%     temptaxis{9}/60,meantimetrace{9},...
%     temptaxis{12}/60,meantimetrace{12})
% ylim([0 20])
% xlabel('time (min)','FontSize',18)
% ylabel('pO_2 (torr)','FontSize',18)
% legend('No Div','2 Div','3 Div','4 Div','6 Div','9 Div','12 Div')
%     
% %PCA filtered
% for PCs = 2:6
%     for numdiv = [1 2 3 4 6 9 12]
%         load(strcat('G:\Documents\Projects\DynamicEPRI\AnimalExperiments\130916_airmouse\timeaxis_',num2str(numdiv),'div.mat'))
%         if numdiv == 1
%             load(strcat('G:\Documents\Projects\DynamicEPRI\AnimalExperiments\130916_airmouse\full_PCAfiltered_',num2str(PCs),'PCs.mat'))
%             Image = full_PCAfiltered.pO2;
%         elseif numdiv == 12
%             load(strcat('G:\Documents\Projects\DynamicEPRI\AnimalExperiments\130916_airmouse\12div_groupPCAfiltered_',num2str(PCs),'PCs(pO2).mat'))
%             Image = div_PCAfiltered_pO2;
%         else
%             load(strcat('G:\Documents\Projects\DynamicEPRI\AnimalExperiments\130916_airmouse\',num2str(numdiv),'div_groupPCAfiltered_',num2str(PCs),'PCs.mat'))
%             Image = div_PCAfiltered.pO2;
%         end
%         
%         x = 31; y = 39; z = 32;
%         timetrace = zeros(27,size(Image,4));
%         count = 0;
%         for ii = -1:1
%             for jj = -1:1
%                 for kk = -1:1
%                     count = count + 1;
%                     timetrace(count,:) = squeeze(Image(x+ii,y+jj,z+kk,:));
%                 end
%             end
%         end
%         meantimetrace{numdiv} = mean(timetrace,1);
%         temptaxis{numdiv} = timeaxis;
%     end
%     figure
%     plot(temptaxis{1}/60,meantimetrace{1},...
%         temptaxis{2}/60,meantimetrace{2},...
%         temptaxis{3}/60,meantimetrace{3},...
%         temptaxis{4}/60,meantimetrace{4},...
%         temptaxis{6}/60,meantimetrace{6},...
%         temptaxis{9}/60,meantimetrace{9},...
%         temptaxis{12}/60,meantimetrace{12})
%     ylim([0 20])
%     xlabel('time (min)','FontSize',18)
%     ylabel('pO_2 (torr)','FontSize',18)
%     legend('No Div','2 Div','3 Div','4 Div','6 Div','9 Div','12 Div')
%     title(strcat(num2str(PCs),'PCs'),'FontSize',24)
% end
% 
% 
% numdiv = 1;
% load(strcat('G:\Documents\Projects\DynamicEPRI\AnimalExperiments\130916_airmouse\timeaxis_',num2str(numdiv),'div.mat'))
% x = 31; y = 39; z = 32;
% timetrace = zeros(27,size(Image,4));
% count = 0;
% for ii = -1:1
%     for jj = -1:1
%         for kk = -1:1
%             count = count + 1;
%             timetrace(count,:) = squeeze(Image(x+ii,y+jj,z+kk,:));
%         end
%     end
% end
% meantimetrace = mean(timetrace,1);
% stdtimetrace = std(timetrace,1);
% errorbar(timeaxis/60,meantimetrace,stdtimetrace,'-o','LineWidth',2)
% xlabel('time (min)','FontSize',18)
% ylabel('pO_2 (torr)','FontSize',18)
% ylim([0 20])