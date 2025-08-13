function tempForDisplay = imagesc3d(image,bounds,dim2);


dims = size(image);

if (length(dims) < 3)
    if (nargin==1)
        imagesc(image);
    else
        imagesc(image,bounds);
    end
    tempForDisplay = image;
else
    numImages = dims(3);

    if (nargin < 3)
        dim2 = ceil(sqrt(numImages));
    end
        dim1 = ceil(numImages/dim2);

    tempForDisplay = zeros(dim1*dims(1),dim2*dims(2));

    for i = 1:numImages
        tempForDisplay(floor((i-1)/dim2)*dims(1)+[1:dims(1)],mod(i-1,dim2)*dims(2)+[1:dims(2)]) = image(:,:,i);
    end
        
    if (nargin==1 || not(prod(size(bounds))))
        imagesc(tempForDisplay);
    else
        imagesc(tempForDisplay,bounds);
    end

    for i = 1:dim1+1
        line([0.5,dim2*dims(2)+0.5],[(i-1)*dims(1)+0.5,(i-1)*dims(1)+0.5]);
    end
    for i = 1:dim2+1
        line([(i-1)*dims(2)+0.5,(i-1)*dims(2)+0.5],[0.5,dim1*dims(1)+0.5]);
    end
end