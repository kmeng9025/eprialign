function [output] = abm_center_crop_image( input, crop_size )
%abm_center_crop_image   Crop the center of 2D or 3D image.
%  by Alan McMillan (amcmillan@umm.edu)
%
%   [output] = abm_center_crop_image( input, crop_size )
%
%   REQUIRED PARAMETERS:
%   input Is the input data matrix, it can be either a 2D or 3D matrix. If
%         3D, then the image will be cropped as 2D slices along the 3rd
%         dimension.
% 
%   crop_size Is the size to crop to, specified like [crop_size_dim1, crop_size_dim2]
%

im_size = size( input );
if ( length(im_size) == 2 ) % specify 3rd dim size as 1 if really 2d
    im_size(3) = 1;
end

% find the center of the image
dim1_center = floor( (im_size(1)+1)/2 );
dim2_center = floor( (im_size(2)+1)/2 );

% determine over which range we will crop
if ( mod(crop_size(1),2) == 1 )
    clip1_start = dim1_center - floor( (crop_size(1)-1)/2 );
    clip1_end   = dim1_center + floor( (crop_size(1)-1)/2 );
else
    clip1_start = dim1_center - floor( (crop_size(1)-1)/2 );
    clip1_end   = dim1_center + ceil( (crop_size(1)-1)/2 );
end    
if ( mod(crop_size(2),2) == 1 )
    clip2_start = dim2_center - floor( (crop_size(2)-1)/2 );
    clip2_end   = dim2_center + floor( (crop_size(2)-1)/2 );
else
    clip2_start = dim2_center - floor( (crop_size(2)-1)/2 );
    clip2_end   = dim2_center + ceil( (crop_size(2)-1)/2 );   
end

% check for errors
if ( clip1_start <= 0 ), error('Clip size too small, attempted to clip to index %g in dimension 1', clip1_start); end
if ( clip2_start <= 0 ), error('Clip size too small, attempted to clip to index %g in dimension 2', clip2_start); end
if ( clip1_end > im_size(1) ), error('Clip size too big, attempted to clip to index %g in dimension 1', clip1_end); end 
if ( clip2_end > im_size(2) ), error('Clip size too big, attempted to clip to index %g in dimension 2', clip2_end); end 

% crop!
output = zeros( crop_size(1), crop_size(2), im_size(3) );
for zz=1:im_size(3)
    output(:,:,zz) = input( clip1_start:clip1_end, clip2_start:clip2_end, zz );
end
    
