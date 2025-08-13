% [Out1<, Out2, ...>] = LoadFitPars(mat_fit, varargin)
% Load fitting results for pulse experiment
% varargin
%   for mat_fit.Algorithm == 'T2_ExpDecay_No_Offset'
%                            {'Amp','T2','Mask', 'Error'}
%   for mat_fit.Algorithm == 'T2_ExpDecay_No_Offset'
%                            {'Amp','LLW','Mask', 'Error'}
%  Example: [Amp, Error] = LoadFitPars(mat_fit, {'Amp', 'Error'})

% boep, 2009

function varargout = LoadFitPars(mat_fit, varargin)
% defualt args
defauls_args = {'Amp','T2','Mask'};
cw_defauls_args = {'Amp','LLW','Mask'};

if ~isfield(mat_fit, 'P')
  disp('Data contain no fit result!');
  for ii = 1:nargout, varargout{ii} = []; end
else
  switch mat_fit.Algorithm
    case 'T2_ExpDecay_No_Offset'
      mask = mat_fit.FitMask;
      if isfield(mat_fit, 'Mask') && ~isempty(mat_fit.Mask)
        mask = mask & mat_fit.Mask;
      end
      if nargin == 1, out_args = defauls_args; else out_args = varargin{1}; end
      for ii = 1:length(out_args)
        fit_val      = zeros(prod(mat_fit.Size),1);
        switch upper(out_args{ii})
          case 'AMP'
            fit_val(mat_fit.Idx) = mat_fit.P(1,:);
          case 'T2'
            fit_val(mat_fit.Idx) = mat_fit.P(2,:);
          case 'MASK'
            fit_val(mat_fit.Idx(mask)) = 1;
            fit_val = logical(fit_val);
          case 'ERROR_STD' % error (standard deviation)
            fit_val(mat_fit.Idx) = mat_fit.Perr(3,:);
          case 'ERROR'     % normalized error (standard deviation)
            if isfield(mat_fit, 'Perr')
              fit_val(mat_fit.Idx) = mat_fit.Perr(3,:)./mat_fit.P(1,:);
            end
          case 'ERROR_AMP' % error of amplitude
            fit_val(mat_fit.Idx) = mat_fit.Perr(1,:);
          case 'ERROR_T2'  % error of T2
            fit_val(mat_fit.Idx) = mat_fit.Perr(2,:);
        end
        varargout{ii} = reshape(fit_val, mat_fit.Size);
      end
    case 'T1_InvRecovery_3Par'
      mask = mat_fit.FitMask;
      if isfield(mat_fit, 'Mask') && ~isempty(mat_fit.Mask)
        mask = mask & mat_fit.Mask;
      end
      if nargin == 1, out_args = defauls_args; else out_args = varargin{1}; end
      for ii = 1:length(out_args)
        fit_val      = zeros(prod(mat_fit.Size),1);
        switch upper(out_args{ii})
          case 'AMP'
            fit_val(mat_fit.Idx) = mat_fit.P(1,:);
          case 'T1'
            fit_val(mat_fit.Idx) = mat_fit.P(2,:);
          case 'MASK'
            fit_val(mat_fit.Idx(mask)) = 1;
            fit_val = logical(fit_val);
          case 'ERROR_STD' % error (standard deviation)
            fit_val(mat_fit.Idx) = mat_fit.Perr(3,:);
          case 'ERROR'     % normalized error (standard deviation)
            if isfield(mat_fit, 'Perr')
              fit_val(mat_fit.Idx) = mat_fit.Perr(3,:)./mat_fit.P(1,:);
            end
          case 'ERROR_AMP' % error of amplitude
            fit_val(mat_fit.Idx) = mat_fit.Perr(1,:);
          case 'ERROR_T1'  % error of T1
            fit_val(mat_fit.Idx) = mat_fit.Perr(2,:);
        end
        varargout{ii} = reshape(fit_val, mat_fit.Size);
      end
    case 'T2T1_InvRecovery_3Par'
      mask = mat_fit.FitMask;
      if isfield(mat_fit, 'Mask') && ~isempty(mat_fit.Mask)
        mask = mask & mat_fit.Mask;
      end
      if nargin == 1, out_args = defauls_args; else out_args = varargin{1}; end
      for ii = 1:length(out_args)
        fit_val      = zeros(prod(mat_fit.Size),1);
        switch upper(out_args{ii})
          case 'AMP'
            fit_val(mat_fit.Idx) = mat_fit.P(1,:);
          case 'T1'
            fit_val(mat_fit.Idx) = mat_fit.P(2,:);
          case 'T2'
            fit_val(mat_fit.Idx) = mat_fit.P(3,:);
          case 'MASK'
            fit_val(mat_fit.Idx(mask)) = 1;
            fit_val = logical(fit_val);
          case 'ERROR_STD' % error (standard deviation)
            fit_val(mat_fit.Idx) = mat_fit.Perr(4,:);
          case 'ERROR'     % normalized error (standard deviation)
            if isfield(mat_fit, 'Perr')
              fit_val(mat_fit.Idx) = mat_fit.Perr(4,:)./mat_fit.P(1,:);
            end
          case 'ERROR_AMP' % error of amplitude
            fit_val(mat_fit.Idx) = mat_fit.Perr(1,:);
          case 'ERROR_T1'  % error of T1
            fit_val(mat_fit.Idx) = mat_fit.Perr(2,:);
          case 'ERROR_T2'  % error of T2
            fit_val(mat_fit.Idx) = mat_fit.Perr(3,:);
        end
        varargout{ii} = reshape(fit_val, mat_fit.Size);
      end
    case 'CW_spectral_fit_R2_XOVR_PHASE'
      if isfield(mat_fit, 'Mask') && ~isempty(mat_fit.Mask)
        mask = mat_fit.Mask;
      else
        mask = 1:length(mat_fit.Idx);
      end
      if nargin == 1, out_args = defauls_args; else out_args = varargin{1}; end
      for ii = 1:length(out_args)
        fit_val      = zeros(prod(mat_fit.Size),1);
        switch upper(out_args{ii})
          case 'AMP'
            fit_val(mat_fit.Idx) = mat_fit.P(1,:);
          case 'XOVER'
            fit_val(mat_fit.Idx) = mat_fit.P(2,:);
          case 'LLW'
            fit_val(mat_fit.Idx) = mat_fit.P(3,:);
          case 'PHASE'
            fit_val(mat_fit.Idx) = mat_fit.P(4,:);
          case 'MASK'
            fit_val(mat_fit.Idx(mask)) = 1;
            fit_val = logical(fit_val);
          case 'ERROR'
            fit_val(mat_fit.Idx) = mat_fit.Perr(5,:);
        end
        varargout{ii} = reshape(fit_val, mat_fit.Size);
      end
    case 'Rabi_sin_2Par'
      mask = mat_fit.FitMask;
      if isfield(mat_fit, 'Mask') && ~isempty(mat_fit.Mask)
        mask = mask & mat_fit.Mask;
      end
      if nargin == 1, out_args = defauls_args; else out_args = varargin{1}; end
      for ii = 1:length(out_args)
        fit_val      = zeros(prod(mat_fit.Size),1);
        switch upper(out_args{ii})
          case 'AMP'
            fit_val(mat_fit.Idx) = mat_fit.P(1,:);
          case 'TP'
            fit_val(mat_fit.Idx) = mat_fit.P(2,:);
          case 'MASK'
            fit_val(mat_fit.Idx(mask)) = 1;
            fit_val = logical(fit_val);
          case 'ERROR_STD' % error (standard deviation)
            fit_val(mat_fit.Idx) = mat_fit.Perr(3,:);
          case 'ERROR'     % normalized error (standard deviation)
            if isfield(mat_fit, 'Perr')
              fit_val(mat_fit.Idx) = mat_fit.Perr(3,:)./mat_fit.P(1,:);
            end
          case 'ERROR_AMP' % error of amplitude
            fit_val(mat_fit.Idx) = mat_fit.Perr(1,:);
          case 'ERROR_T1'  % error of T1
            fit_val(mat_fit.Idx) = mat_fit.Perr(2,:);
        end
        varargout{ii} = reshape(fit_val, mat_fit.Size);
      end      
  end
end