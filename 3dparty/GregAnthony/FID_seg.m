%Fiducial Segmentation
%Greg Anthony
%2015

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [FID] = FID_seg(OtsuMask)

FID = OtsuMask;
for i=1:size(FID,3)
FID(:,:,i) = imfill(FID(:,:,i),'holes');
end

%Label Regions
FIDLabel = zeros(size(OtsuMask));
numObjects = zeros(size(OtsuMask,3),1);

for i=1:size(FID,3)
    [FIDLabel(:,:,i),numObjects(i,1)] = bwlabel(FID(:,:,i)); 
    Stats{i} = regionprops(FIDLabel(:,:,i));
end

%Remove regions too small/big or too close to center
for i=1:size(FID,3)
    Slice = FIDLabel(:,:,i);
    for n=1:numObjects(i,1)
        if Stats{i}(n).Area<50 || Stats{i}(n).Area>500
            Slice(Slice==n)=0;
        elseif Stats{i}(n).Centroid(1)<(size(OtsuMask,2)/2)+40 && Stats{i}(n).Centroid(1)>(size(OtsuMask,2)/2)-40 && Stats{i}(n).Centroid(2)<(size(OtsuMask,1)/2)+40 && Stats{i}(n).Centroid(2)>(size(OtsuMask,1)/2)-40
            Slice(Slice==n)=0;
        else 
            Slice(Slice==n)=1;
        end   
    end
    FID(:,:,i) = Slice;
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%