% EPRI_DECODEIMAGETYPE  convert type string to FBP structure
% FBP = epri_DecodeImageType(type_string)
% type_string - sequence of parameters each of them 1-2 letters followed by
% a value [string]. Parameters order and spaces are ignored
% Example: SOP T2 image is SB18T14S1B4G1.5Z0P3
%          MB [-1 0 1]     MB18T14S1B4G1.5Z0P3MA1MS3MF1
%
% For single B0
%   SB - spatial angles
%   A -  spectral angles
%   G - maximum gradient [G/cm]
%   B - baseline type
%   Z - zero gradient type
%   S - image sampling type
%   T - image type
%   P - coordinate pole
%   O - order, 0 - default, 1 - MSPS [(0)/1]
%   D - overall number of data files
%   N - Navigator projections
%
% For multi B0
%   MB - angles
%   G  - maximum gradient [G/cm]
%   B  - baseline type
%   Z  - zero gradient type
%   S  - image sampling type
%   T  - image type
%   P  - coordinate pole
%   MA - algorithm
%   MS - number of steps
%   MF - maximum offset [G]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, 2013
% Contact: epri.uchicago.edu

function IMAGE_STRUCTURE =epri_DecodeImageType(image_type)
image_type = strtrim(image_type);

if contains(image_type, 'SB') ||  contains(image_type, 'MB')  ||  contains(image_type, 'RS')
  FBP = [];
  MB0 =[];
  
  FBP.nSpec = 1;
  FBP.msps = 'off';
  FBP.baseline = 'none';
  FBP.zero_gradient = 'none';
  FBP.angle_sampling = 'uniform_spatial_flip';
  FBP.nData = 1;
  
  if contains(image_type, 'SB')
    FBP.scheme = 'single_b';
    k = regexp(image_type(1:end), '(?<par>[A-Z]+)(?<val>[-+0-9.,]+)', 'names');
    for ii = 1:length(k)
      switch(k(ii).par)
        case 'SB', FBP.nPolar = str2double(k(ii).val); FBP.nAz = FBP.nPolar;
        case 'A', FBP.nSpec = str2double(k(ii).val);
        case 'G', FBP.MaxGradient = str2double(k(ii).val);
        case 'B', idx = fix(str2double(k(ii).val));
          if idx == 0, FBP.baseline = 'none';
          elseif idx > 0, FBP.baseline = 'every_n'; FBP.bl_n = cast(idx, 'double');
          elseif idx == -1, FBP.baseline = 'before';
          elseif idx == -2, FBP.baseline = 'after';
          elseif idx == -3, FBP.baseline = 'before_after';
          end
        case 'Z', idx = fix(str2double(k(ii).val));
          if idx == 0, FBP.zero_gradient = 'none';
          elseif idx > 0, FBP.zero_gradient = 'every_n'; FBP.zg_n = cast(idx, 'double');
          elseif idx == -1, FBP.zero_gradient = 'before';
          elseif idx == -2, FBP.zero_gradient = 'after';
          elseif idx == -3, FBP.zero_gradient = 'before_after';
          end
        case 'S', idx = fix(str2double(k(ii).val));
          if idx == 1, FBP.angle_sampling = 'uniform_spatial_flip';
          elseif idx == 2, FBP.angle_sampling = 'uniform_angular_flip';
          elseif idx == 5, FBP.angle_sampling = 'equal_solid_angle_gage';
          elseif idx == 6, FBP.angle_sampling = 'UNIFORM_SPATIAL_SPIRAL_8Q';
          end
        case 'O', val = fix(str2double(k(ii).val));
          if val == 0, FBP.projection_order = 'default';
          else FBP.projection_order = 'msps';
          end
        case 'D', val = fix(str2double(k(ii).val));
          FBP.nData = val;
        case 'T', FBP.imtype = fix(str2double(k(ii).val));
        case 'P', idx = fix(str2double(k(ii).val));
          if idx == 1, FBP.CoordPole = 'X';
          elseif idx == 2, FBP.CoordPole = 'Y';
          elseif idx == 3, FBP.CoordPole = 'Z';
          end
        case 'N', idx = fix(str2double(k(ii).val));
          if idx > 0
            FBP.NAV.type = 'all_delay';
          else
            FBP.NAV.type = 'none';
          end
        case 'NN'
          FBP.NAV.n = fix(str2double(k(ii).val));
      end
    end
  elseif contains(image_type, 'MB')
    FBP.scheme = 'multi_b';
    k = regexp(image_type(1:end), '(?<par>[A-Z]+)(?<val>[-+0-9.,]+)', 'names');
    for ii = 1:length(k)
      switch(k(ii).par)
        case 'MB', FBP.nPolar = str2double(k(ii).val); FBP.nAz = FBP.nPolar;
        case 'G', FBP.MaxGradient = str2double(k(ii).val);
        case 'B', idx = fix(str2double(k(ii).val));
          if idx == 0, FBP.baseline = 'none';
          elseif idx > 0, FBP.baseline = 'every_n'; FBP.bl_n = cast(idx, 'double');
          elseif idx == -1, FBP.baseline = 'before';
          elseif idx == -2, FBP.baseline = 'after';
          elseif idx == -3, FBP.baseline = 'before_after';
          end
        case 'Z', idx = fix(str2double(k(ii).val));
          if idx == 0, FBP.zero_gradient = 'none';
          elseif idx > 0, FBP.zero_gradient = 'every_n'; FBP.zg_n = cast(idx, 'double');
          elseif idx == -1, FBP.zero_gradient = 'before';
          elseif idx == -2, FBP.zero_gradient = 'after';
          elseif idx == -3, FBP.zero_gradient = 'before_after';
          end
        case 'S', idx = fix(str2double(k(ii).val));
          if idx == 1, FBP.angle_sampling = 'uniform_spatial_flip';
          elseif idx == 1, FBP.angle_sampling = 'uniform_angular_flip';
          end
        case 'T', FBP.imtype = fix(str2double(k(ii).val));
        case 'P', idx = fix(str2double(k(ii).val));
          if idx == 1, FBP.CoordPole = 'X';
          elseif idx == 2, FBP.CoordPole = 'Y';
          elseif idx == 3, FBP.CoordPole = 'Z';
          end
        case 'MA', MB0.MBadaptive = fix(str2double(k(ii).val));
        case 'MS', MBsteps = fix(str2double(k(ii).val));
        case 'MF', Fmax = str2double(k(ii).val);
          MB0.Offsets = -Fmax:2*Fmax/(MBsteps-1):Fmax;
      end
    end
  elseif contains(image_type, 'RS')
    FBP.scheme = 'rapid_scan';
    k = regexp(image_type(1:end), '(?<par>[A-Z]+)(?<val>[-+0-9.,]+)', 'names');
    for ii = 1:length(k)
      switch(k(ii).par)
        case 'RS', FBP.nPolar = str2double(k(ii).val); FBP.nAz = FBP.nPolar;
        case 'A', FBP.nSpec = str2double(k(ii).val);
        case 'G', FBP.MaxGradient = str2double(k(ii).val);
        case 'B', idx = fix(str2double(k(ii).val));
          if idx == 0, FBP.baseline = 'none';
          elseif idx > 0, FBP.baseline = 'every_n'; FBP.bl_n = cast(idx, 'double');
          elseif idx == -1, FBP.baseline = 'before';
          elseif idx == -2, FBP.baseline = 'after';
          elseif idx == -3, FBP.baseline = 'before_after';
          end
        case 'Z', idx = fix(str2double(k(ii).val));
          if idx == 0, FBP.zero_gradient = 'none';
          elseif idx > 0, FBP.zero_gradient = 'every_n'; FBP.zg_n = cast(idx, 'double');
          elseif idx == -1, FBP.zero_gradient = 'before';
          elseif idx == -2, FBP.zero_gradient = 'after';
          elseif idx == -3, FBP.zero_gradient = 'before_after';
          end
        case 'S', idx = fix(str2double(k(ii).val));
          if idx == 1, FBP.angle_sampling = 'uniform_spatial_flip';
          elseif idx == 2, FBP.angle_sampling = 'uniform_angular_flip';
          elseif idx == 5, FBP.angle_sampling = 'equal_solid_angle_gage';
          end
        case 'ST', val = fix(str2double(k(ii).val));
          FBP.nTheta = val;
        case 'SP', val = fix(str2double(k(ii).val));
          FBP.nPhi = val;
        case 'T', FBP.imtype = fix(str2double(k(ii).val));
        case 'P', idx = fix(str2double(k(ii).val));
          if idx == 1, FBP.CoordPole = 'X';
          elseif idx == 2, FBP.CoordPole = 'Y';
          elseif idx == 3, FBP.CoordPole = 'Z';
          end
        case 'MA', FBP.baseline = 'split_field';
        case 'MS', MBsteps = fix(str2double(k(ii).val));
        case 'MF', Fmax = str2double(k(ii).val);
          FBP.split_field = -Fmax:2*Fmax/(MBsteps-1):Fmax;
      end
    end
  else
    IMAGE_STRUCTURE = [];
    return;
  end
  FBP.MB0 = MB0;
  FBP.Order = safeget(FBP, 'Order', 0);
  IMAGE_STRUCTURE = FBP;
  % ------SPI------
elseif contains(image_type, 'SP')
  SPI = [];
    
  SPI.Method = '1';
  SPI.nSteps = [0,0,0];
  k = regexp(image_type(1:end), '(?<par>[A-Z]+)(?<val>[-+0-9.,]+)', 'names');
  for ii = 1:length(k)
    switch(k(ii).par)
      case 'SP', SPI.nSteps(3) = str2double(k(ii).val); 
      case 'X', SPI.nSteps(1) = str2double(k(ii).val); 
      case 'Y', SPI.nSteps(2) = str2double(k(ii).val); 
      case 'A',  FBP.nSpec = str2double(k(ii).val);
      case 'G',  SPI.MaxGradient = str2double(k(ii).val);
      case 'B', idx = fix(str2double(k(ii).val));
        if idx == 0, SPI.baseline = 'none';
        elseif idx > 0, SPI.baseline = 'every_n'; SPI.bl_n = cast(idx, 'double');
        elseif idx == -1, SPI.baseline = 'before';
        elseif idx == -2, SPI.baseline = 'after';
        elseif idx == -3, SPI.baseline = 'before_after';
        end
      case 'Z', idx = fix(str2double(k(ii).val));
        if idx == 0, SPI.zero_gradient = 'none';
        elseif idx > 0, SPI.zero_gradient = 'every_n'; SPI.zg_n = cast(idx, 'double');
        elseif idx == -1, SPI.zero_gradient = 'before';
        elseif idx == -2, SPI.zero_gradient = 'after';
        elseif idx == -3, SPI.zero_gradient = 'before_after';
        end
      case 'S', idx = fix(str2double(k(ii).val));
        switch idx
          case 1, SPI.Method = '8Q_no_optimization';
          case 2, SPI.Method = '8Q_flip_optimization';
          case 3, SPI.Method = '4Q_flip_optimization';
          case 4, SPI.Method = 'Full_Rectangular';
        end
      case 'N', idx = fix(str2double(k(ii).val));
        if idx > 0
          FBP.NAV.type = 'all_delay';
        else
          FBP.NAV.type = 'none';
        end
      case 'NN'
        FBP.NAV.n = fix(str2double(k(ii).val));
    end
  end
  if SPI.nSteps(1)==0, SPI.nSteps(1)=SPI.nSteps(3); end
  if SPI.nSteps(2)==0, SPI.nSteps(2)=SPI.nSteps(3); end
  IMAGE_STRUCTURE = SPI;
end