% [sizeReson, mask] = epr_GetResonatorSize(resonType, matrix_size, FOV)
% resonType - '16mm'/'19mm'/'1inchSlotted'/'51.1x57.2mm'
% matrix_size and FOV - optional parameters for the mask generation
% mask - optional, mask of resonator volume
% Example: [res_dims, res_mask] = epr_GetResonatorSize('19mm', 64, 4.24);

function [sizeReson, varargout] = epr_GetResonatorSize(resonType, matrix_size, FOV)

switch resonType % D, L in cm
  case '16mm'
    Dcm=1.6; Lcm=1.5;
  case '19mm'
    Dcm=1.9; Lcm=1.5;
  case '1inchSlotted'
    Dcm=2.58; Lcm=3.20;
  case '51.1x57.2mm'
    Dcm=5.114; Lcm=5.725;
  otherwise
    Dcm = 0; Lcm = 0;
end
sizeReson = [Dcm; Lcm];

if nargout > 1
  if ~exist('matrix_size', 'var'), matrix_size = 64; end
  if ~exist('FOV', 'var'), FOV = 3*sqrt(2); end
  resonator = zeros(matrix_size*[1,1,1]);
  center = (matrix_size+1)/2;
  dim = FOV*((1:matrix_size)'-center)/(matrix_size-1);
  dim1 = dim(:, ones(matrix_size, 1), ones(matrix_size, 1)); %x
  dim2 = permute(dim1, [2,1,3]); %y
  dim3 = permute(dim1, [2,3,1]); %z
  
  resonator(dim2 <= Lcm/2 & dim2 >= -Lcm/2 & (dim3.^2+dim1.^2) < Dcm^2/4) = 1;
  varargout{1} = resonator;
end
% END OF function GetSizeResonator
