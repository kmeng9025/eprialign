function [image_final, xp, yp] = resample_plane(plane_data, tcorners, nPtsx, nPtsy)  

data_class = class(plane_data);
nplanes = 1;
sz=size(plane_data);
if (numel(sz) > 2), 
   nplanes = size(plane_data, 3);
end
nanval = 0;
image_final = zeros(nPtsy, nPtsx, nplanes, data_class);

[xp,yp]=generate_slice_coordinates(tcorners, nPtsx, nPtsy);

if (nplanes > 1),
    for n = 1:nplanes
        myplane = squeeze(plane_data(:,:,n));
        if strcmp(data_class,'uint8') 
            myplane = cast(myplane, 'double');
            image_final(:,:,n) = cast(interp2(myplane, xp, yp,'linear',nanval), data_class);
        else
            image_final(:,:,n) = interp2(myplane, xp, yp,'linear',nanval);
        end
    end
else
    if strcmp(data_class,'uint8') 
        myplane = cast(plane_data, 'double');
        iplane=interp2(myplane, xp, yp,'linear',nanval);
        image_final = cast(iplane, data_class);
    else
        image_final = interp2(plane_data, xp, yp,'linear',nanval);
    end
end
