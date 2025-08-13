%Leg Segmentation
%Greg Anthony
%2015

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Leg] = Leg_seg(OtsuMask,FID)

Leg = OtsuMask;
SE = strel('disk',5,0);
for i=1:size(OtsuMask,3) 
    %Remove fiducials
    Leg(FID==1)=0;
    %Dilate
    Leg(:,:,i) = imdilate(Leg(:,:,i),SE);
    %Fill in holes/inlets
    Leg(:,:,i) = imfill(Leg(:,:,i),'holes');
    %Erode
    Leg(:,:,i) = imerode(Leg(:,:,i),SE);
end

LegLabel = zeros(size(OtsuMask));
numObjects = zeros(size(OtsuMask,3),1);

%Label image regions
for i=1:size(Leg,3)
    [LegLabel(:,:,i),numObjects(i,1)] = bwlabel(Leg(:,:,i)); 
    Stats{i} = regionprops(LegLabel(:,:,i));
end

for i=1:size(OtsuMask,3)
    Slice = LegLabel(:,:,i);
%Keep only the largest region
    if numObjects(i,1)>1
        big = 1;
        for n=2:numObjects(i,1)
            if Stats{i}(n).Area > Stats{i}(big).Area
                Slice(Slice==big)=0;
                Slice(Slice==n)=1;
                big = n;
            else
                Slice(Slice==n)=0;
                Slice(Slice==big)=1;
            end
        end
    end
    Leg(:,:,i) = Slice;
end

end
