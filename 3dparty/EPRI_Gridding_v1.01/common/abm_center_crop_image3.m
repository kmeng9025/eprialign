function [output] = abm_center_crop_image3( input, crop_size )
%abm_center_crop_image   Crop the center of 3D image.

if ( length(size(input)) ~= 3 ) 
	error('Input data must be 3 dimensional.')
end

if ( length(crop_size) ~= 3 ) 
	error('Crop size must be 3 dimensional.')
end

% find the center of the image
dim1_center = floor( (size(input,1)+1)/2 );
dim2_center = floor( (size(input,2)+1)/2 );
dim3_center = floor( (size(input,3)+1)/2 );

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
if ( mod(crop_size(2),2) == 1 )
    clip3_start = dim2_center - floor( (crop_size(3)-1)/2 );
    clip3_end   = dim2_center + floor( (crop_size(3)-1)/2 );
else
    clip3_start = dim2_center - floor( (crop_size(3)-1)/2 );
    clip3_end   = dim2_center + ceil( (crop_size(3)-1)/2 );   
end

% check for errors
if ( clip1_start <= 0 ), error('Clip size too small, attempted to clip to index %g in dimension 1', clip1_start); end
if ( clip2_start <= 0 ), error('Clip size too small, attempted to clip to index %g in dimension 2', clip2_start); end
if ( clip3_start <= 0 ), error('Clip size too small, attempted to clip to index %g in dimension 3', clip3_start); end

if ( clip1_end > size(input,1) ), error('Clip size too big, attempted to clip to index %g in dimension 1', clip1_end); end 
if ( clip2_end > size(input,2) ), error('Clip size too big, attempted to clip to index %g in dimension 2', clip2_end); end 
if ( clip3_end > size(input,3) ), error('Clip size too big, attempted to clip to index %g in dimension 3', clip3_end); end 

% crop!
output = input( clip1_start:clip1_end, clip2_start:clip2_end, clip3_start:clip3_end );

    
