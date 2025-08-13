function [mat_recFXD, fields, dsc] = epri_reconstruct(in_y, mat_info, fields, varargin)
% function [ax, y, dsc] = td_reconstruct(in_y, mat_info, mat_rec_info, ...)
% Pulse EPRI image reconstruction
%   in_opt.phase_algorithm   - algorithm of phase optimization
%        'manual_zero_order' - rotate on angle in_opt.phase_zero_order
%        'max_real_single'   - indepenent, trace by trace
%        'max_real_all'      - use one phase for all slices
%   ...                      - parameter-value comma separated pairs
%   out_pars                 - structure of supplementary information

% -------------------------------------------------------------------------
% --------------  parameters number checking   ----------------------------
% -------------------------------------------------------------------------
if nargin < 3
  error('Usage: [out_y, out_pars] = td_reconstruct(in_y, in_opt, mat_info, mat_rec_info, vargin)');
elseif nargin > 3
  if ~mod(nargin-1,2)
    for kk=1:2:nargin-1
      in_opt=setfield(in_opt, lower(varargin{kk}), varargin{kk+1});
    end
  else error('td_reconstruct: Wrong amount of arguments')
  end
end

rec = fields.rec;
fbp = fields.fbp;

% reconstruction parameter
if safeget(rec, 'DoublePoints', 0)
  Sub_points = rec.Sub_points * 2;
else
  Sub_points = rec.Sub_points;
end

CodeFlag = safeget(rec, 'CodeFlag', 'MATLAB'); %%% select a method to reconstruct
mat_recFXD = zeros(Sub_points,Sub_points,Sub_points,size(in_y,2));%%% define a array to store the final image

tic; % Record the reconstruction time

% normalize amplitude on the unit volume/1D
switch 14
  case 14, n = 3; % 3D experiment
end

sz = size(in_y);
pidx = safeget(rec, 'projection_index', []);
    
switch CodeFlag
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  case {'SINGLEv1', 'SINGLEv1 GPU'}
    if ~isempty(pidx)
      in_y = in_y(:,:,pidx);
      Pars.GradX = mat_info.GradX(pidx);
      Pars.GradY = mat_info.GradY(pidx);
      Pars.GradZ = mat_info.GradZ(pidx);
    else
      Pars.GradX = mat_info.G(:,1);
      Pars.GradY = mat_info.G(:,2);
      Pars.GradZ = mat_info.G(:,3);
    end
    Pars.nDim = 3;
    Rec.Filter = safeget(rec,'Filter','ram-lak');
    Rec.CutOff = fields.rec.FilterCutOff;
    Rec.nBins = Sub_points;
    Rec.Intrp = safeget(rec,'Intrp','TriIDW');
    Rec.Process = iff(strcmp(CodeFlag, 'SINGLEv1 GPU'), 'GPU', 'CPU');
    
    Rec.Directions = 'Gradients';
    for n_echo=1:sz(2)
      p = squeeze(in_y(:,n_echo,:));
      [FXD] = Reconstruct(p,Pars,Rec);
      
      if safeget(rec, 'DoublePoints', 0)
        switch n
          case 3
            FXD = reshape(FXD, [2,rec.Sub_points,2,rec.Sub_points,2,rec.Sub_points]);
            FXD =  squeeze(sum(sum(sum(FXD, 1), 3), 5))/8;
        end
      end
      mat_recFXD(:,:,:,n_echo) = FXD;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  case {'SINGLEv2', 'SINGLEv2 GPU'}
    recon_pars.nBins = Sub_points;
    recon_pars.FilterCutOff = Sub_points/sz(1)*rec.FilterCutOff; % [0-1] full bandwidth of projections
    recon_pars.FilterCutOff = min(recon_pars.FilterCutOff, 1);
    recon_pars.size = rec.Size;
    recon_pars.display=0;
    % 2 means 3-points method, which is faster and has a good compromise.
    % 4 means Ram-Lak filter only
    recon_pars.Filter=2;
    recon_pars.processor=iff(strcmp(CodeFlag, 'SINGLEv2 GPU'), 2, 1);     % 1 is for cpu; 2 is for GPU
    recon_pars.interp_method=1; % 0 is zero-rank interpolation method; 1 is linear interpolaiton method,2 is the spline interpolation.
    recon_pars.tasksliced=1;    % [0/1] use 1 to break task in multiple executions
    
    % radon parameters
    [~, pars_ext] = iradon_FBPGradTable(fbp);
    
    if ~isempty(pidx)
      in_y = in_y(:,:,pidx);
      radon_pars.x = pars_ext.kx(pidx);
      radon_pars.y = pars_ext.ky(pidx);
      radon_pars.z = pars_ext.kz(pidx);
      % enforce weight calculations
      radon_pars.w = iradon_vor_area_3d(radon_pars.x,radon_pars.y,radon_pars.z);
    else
      radon_pars.x = pars_ext.kx;
      radon_pars.y = pars_ext.ky;
      radon_pars.z = pars_ext.kz;
      radon_pars.w = pars_ext.w;
    end
    radon_pars.size = rec.Size;
    
    for n_echo=1:sz(2)
      FXD = iradon_3d_sstage_v2(squeeze(in_y(:,n_echo,:)),radon_pars,recon_pars);
      
      if safeget(rec, 'DoublePoints', 0)
        switch n
          case 3
            FXD = reshape(FXD, [2,rec.Sub_points,2,rec.Sub_points,2,rec.Sub_points]);
            FXD =  squeeze(sum(sum(sum(FXD, 1), 3), 5))/8;
        end
      end
      mat_recFXD(:,:,:,n_echo) = FXD;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  otherwise % 'MATLAB' others do not work
    if ~isempty(pidx)
      in_y1 = zeros(size(in_y));
      in_y1(:,:,pidx) = in_y(:,:,pidx);
      in_y = in_y1;
    end
    % permutation of projections
    yy = zeros(size(in_y));
    yy(:, :, mat_info.gidx) = real(single(in_y));
    sz = size(yy);

    angle_sampling = upper(safeget(fbp,'angle_sampling','uniform_spatial'));

    pre_resize_matrix_dims = [sz(1),prod(sz(3:end))];
    recon_matrix_dims = [Sub_points,1,fbp.nAz,fbp.nPolar];
    post_resize_matrix_dims = [recon_matrix_dims(1), prod(recon_matrix_dims(2:end))];
    for n_echo=1:sz(2)
      % sub/supersampling
      single_tau_projections = squeeze(yy(:,n_echo,:));
      switch 'imresize'
        case 'imresize'
          mat_out = imresize(single_tau_projections, post_resize_matrix_dims);
        case 'interpft'
          mat_out = interpft(single_tau_projections, Sub_points, 1);
        case 'spline'
          yyy = reshape(single_tau_projections, pre_resize_matrix_dims);
          for ii = 1 : prod(sz(2:end))
            mat_out(:, ii) = spline(1:sz(1), yyy(:, ii), linspace(1, sz(1), Sub_points));
          end
        case 'gaussian' % % gaussian convolution
          % mat_out=subsampleimagedata(y,rec.Sub_points,length(t)/rec.Sub_points,1);
        otherwise
          mat_out = single_tau_projections;
      end
      mat_out = reshape(mat_out, recon_matrix_dims);

      switch angle_sampling
        case {'UNIFORM_SPATIAL','UNIFORM_SPATIAL_FLIP'}
          mat_out=iradon_InterpToUniformAngle(mat_out,'imgData');
      end

      % Reconstruction
      radon_pars.ELA =  fbp;
      radon_pars.size = rec.Size;
      recon_pars = rec;
      FXD = iradon_d2d_mstage(mat_out, radon_pars, recon_pars);

      if safeget(rec, 'DoublePoints', 0)
        switch n
          case 3
            FXD = reshape(FXD, [2,rec.Sub_points,2,rec.Sub_points,2,rec.Sub_points]);
            FXD =  squeeze(sum(sum(sum(FXD, 1), 3), 5))/8;
        end
      end
      mat_recFXD(:,:,:,n_echo) = FXD;
    end
    disp('    Reconstruction is finished.');
end
toc

% scale  = rec.Size/rec.Sub_points/2;
% ax.x = scale*[-rec.Sub_points+1:2:rec.Sub_points]';
% ax.y = scale*[-rec.Sub_points+1:2:rec.Sub_points]';
% ax.z = scale*[-rec.Sub_points+1:2:rec.Sub_points]';
% ax.xlabel = 'l,cm';
% ax.ylabel = 'l,cm';
% ax.zlabel = 'l,cm';

dsc.rec = rec;
