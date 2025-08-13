%% return slice with current set values
% caution: this function is not effective to deal with data with large matrix.
function img2D = getSlice(img3D, axis, pxSlice, T)

if ~exist('T')
    if axis == 1 % x-axis
        img2D = squeeze(img3D(:,:,pxSlice));
    elseif axis == 2 % y-axis
        img2D = squeeze(img3D(pxSlice,:,:));    
    elseif axis == 3 % z-axis
        img2D = squeeze(img3D(:,pxSlice,:));    
    end
else
    if axis == 1 % x-axis
        img2D = squeeze(img3D(:,:,pxSlice,T));
    elseif axis == 2 % y-axis
        img2D = squeeze(img3D(pxSlice,:,:,T));    
    elseif axis == 3 % z-axis
        img2D = squeeze(img3D(:,pxSlice,:,T));    
    end    
end

end

