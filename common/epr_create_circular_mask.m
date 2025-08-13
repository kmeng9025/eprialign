function [ Circular_mask ] = epr_create_circular_mask( Mask_size , Diameter )
%often I find myself writting a few lines to generate a 2-D circle. This
%code should allow me to generalize that process.make sure Diameter is in
%pixles and fits within mask_size.

%MM 12/27/2016

Circular_mask = zeros(Mask_size);
[X,Y] = meshgrid(1:Mask_size(1),1:Mask_size(2));
dist = sqrt((X-round(Mask_size(1)/2)).^2 + (Y-round(Mask_size(2)/2)).^2);

Circular_mask(dist <=(Diameter/2)) =1;

end

