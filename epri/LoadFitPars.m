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

if ~isfield(mat_fit, 'P')
  disp('Data contain no fit results!');
  for ii = 1:nargout, varargout{ii} = []; end
else
  switch mat_fit.Algorithm
    case 'T2_ExpDecay_No_Offset'
      iAMP = 1; iT2 = 2; iERR = 3;
      mask = mat_fit.FitMask;
      if isfield(mat_fit, 'Mask') && ~isempty(mat_fit.Mask)
        mask = mask & mat_fit.Mask;
      end
      if nargin == 1, out_args = defauls_args; else out_args = varargin{1}; end
      for ii = 1:length(out_args)
        fit_val      = zeros(prod(mat_fit.Size),1);
        switch upper(out_args{ii})
          case 'AMP'
            fit_val(mat_fit.Idx) = mat_fit.P(iAMP,:);
          case 'T2'
            fit_val(mat_fit.Idx) = mat_fit.P(iT2,:);
          case 'MASK'
            fit_val(mat_fit.Idx(mask)) = 1;
            fit_val = logical(fit_val);
          case 'ERROR_STD' % error (standard deviation)
            fit_val(mat_fit.Idx) = mat_fit.Perr(iERR,:);
          case 'ERROR'     % normalized error (standard deviation)
            if isfield(mat_fit, 'Perr')
              fit_val(mat_fit.Idx) = mat_fit.Perr(iERR,:)./mat_fit.P(iAMP,:);
            end
          case 'ERROR_AMP' % error of amplitude
            fit_val(mat_fit.Idx) = mat_fit.Perr(iAMP,:)./mat_fit.P(iAMP,:);
          case 'ERROR_R2'  % error of R2
            fit_val(mat_fit.Idx) = mat_fit.Perr(iT2,:);
        end
        varargout{ii} = reshape(fit_val, mat_fit.Size);
      end
    case 'T1_InvRecovery_3Par'
      iAMP = 1; iT1 = 2; iINV = 3; iERR = 4;
      mask = mat_fit.FitMask;
      if isfield(mat_fit, 'Mask') && ~isempty(mat_fit.Mask)
        mask = mask & mat_fit.Mask;
      end
      if nargin == 1, out_args = defauls_args; else out_args = varargin{1}; end
      for ii = 1:length(out_args)
        fit_val      = zeros(prod(mat_fit.Size),1);
        switch upper(out_args{ii})
          case 'AMP'
            fit_val(mat_fit.Idx) = mat_fit.P(iAMP,:);
          case 'T1'
            fit_val(mat_fit.Idx) = mat_fit.P(iT1,:);
          case 'R1'
            fit_val(mat_fit.Idx) = 1./mat_fit.P(iT1,:);
          case 'MASK'
            fit_val(mat_fit.Idx(mask)) = 1;
            fit_val = logical(fit_val);
          case 'ERROR_ABS' % error 
            fit_val(mat_fit.Idx) = mat_fit.Perr(iERR,:);
          case 'ERROR'     % normalized error (standard deviation)
            if isfield(mat_fit, 'Perr')
              fit_val(mat_fit.Idx) = mat_fit.Perr(iERR,:)./mat_fit.P(iAMP,:);
            end
          case 'ERROR_AMP' % error of amplitude
            fit_val(mat_fit.Idx) = mat_fit.Perr(iAMP,:);
          case 'ERROR_R1'  % error of R1
            fit_val(mat_fit.Idx) = mat_fit.Perr(iT1,:)./mat_fit.P(iT1,:)./mat_fit.P(iT1,:);
        end
        varargout{ii} = reshape(fit_val, mat_fit.Size);
      end
    case 'T1_InvRecovery_3ParR1'
      iAMP = 1; iR1 = 2; iINV = 3; iERR = 4;
      mask = mat_fit.FitMask;
      if isfield(mat_fit, 'Mask') && ~isempty(mat_fit.Mask)
        mask = mask & mat_fit.Mask;
      end
      if nargin == 1, out_args = defauls_args; else out_args = varargin{1}; end
      for ii = 1:length(out_args)
        fit_val      = zeros(prod(mat_fit.Size),1);
        switch upper(out_args{ii})
          case 'AMP'
            fit_val(mat_fit.Idx) = mat_fit.P(iAMP,:);
          case 'T1'
            fit_val(mat_fit.Idx) = 1./mat_fit.P(iR1,:);
            fit_val(isinf(fit_val)) = 0;
          case 'R1'
            fit_val(mat_fit.Idx) = mat_fit.P(iR1,:);
          case 'MASK'
            fit_val(mat_fit.Idx(mask)) = 1;
            fit_val = logical(fit_val);
          case 'ERROR_ABS' % error 
            fit_val(mat_fit.Idx) = mat_fit.Perr(iERR,:);
          case 'ERROR'     % normalized error (standard deviation)
            if isfield(mat_fit, 'Perr')
              fit_val(mat_fit.Idx) = mat_fit.Perr(iERR,:)./mat_fit.P(iAMP,:);
            end
          case 'ERROR_AMP' % error of amplitude
            fit_val(mat_fit.Idx) = mat_fit.Perr(iAMP,:);
          case 'ERROR_R1'  % error of R1
            fit_val(mat_fit.Idx) = mat_fit.Perr(iR1,:);
        end
        varargout{ii} = reshape(fit_val, mat_fit.Size);
      end
    case 'T2T1_InvRecovery_3Par'
      iAMP = 1; iT1 = 2; iT2 = 3; iINV = 4; iERR = 5;
      mask = mat_fit.FitMask;
      if isfield(mat_fit, 'Mask') && ~isempty(mat_fit.Mask)
        mask = mask & mat_fit.Mask;
      end
      if nargin == 1, out_args = defauls_args; else out_args = varargin{1}; end
      for ii = 1:length(out_args)
        fit_val      = zeros(prod(mat_fit.Size),1);
        switch upper(out_args{ii})
          case 'AMP'
            fit_val(mat_fit.Idx) = mat_fit.P(iAMP,:);
          case 'T1'
            fit_val(mat_fit.Idx) = mat_fit.P(iT1,:);
          case 'T2'
            fit_val(mat_fit.Idx) = mat_fit.P(iT2,:);
          case 'MASK'
            fit_val(mat_fit.Idx(mask)) = 1;
            fit_val = logical(fit_val);
          case 'ERROR_STD' % error (standard deviation)
            fit_val(mat_fit.Idx) = mat_fit.Perr(iERR,:);
          case 'ERROR'     % normalized error (standard deviation)
            if isfield(mat_fit, 'Perr')
              fit_val(mat_fit.Idx) = mat_fit.Perr(iERR,:)./mat_fit.P(iAMP,:);
            end
          case 'ERROR_AMP' % error of amplitude
            fit_val(mat_fit.Idx) = mat_fit.Perr(iAMP,:);
          case 'ERROR_T1'  % error of T1
            fit_val(mat_fit.Idx) = mat_fit.Perr(iT1,:);
          case 'ERROR_T2'  % error of T2
            fit_val(mat_fit.Idx) = mat_fit.Perr(iT2,:);
          case 'ERROR_R1'  % error of R1
            fit_val(mat_fit.Idx) = real(mat_fit.Perr(iT1,:));
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
      case 'max_1Par'
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
          end
          varargout{ii} = reshape(fit_val, mat_fit.Size);
        end
  end
end