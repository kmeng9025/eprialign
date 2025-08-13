%%Automated FSA leg tumor segmentation for axial MRI images of C3H mice%%
%Greg Anthony
%2015

%Before running, run the Center MATLAB toolbox path and export the image 
%sequence to be segmented with the Arbuz GUI. 

function [FID,Leg,Tumor] = FSA_auto_seg(export) 
%Image is a struct path, e.g. export.Images{1}.Image

Image = export.Images{1}.Image;
%LegMask = export.Masks{2}.Mask;
%FIDMask = export.Masks{3}.Mask;
%TumorMask = export.Masks{4}.Mask;
%Leg = LegMask.*Image;
%FID = FIDMask.*Image;
%Tumor = TumorMask.*Image;

%Remove background with Otsu thresholding mask
Threshes = zeros(size(Image,3),1);
NormImage = zeros(size(Image));
OtsuMask = zeros(size(Image));

for i=1:size(Image,3)
    NormImage(:,:,i) = (1/(max(max(Image(:,:,i)))))*Image(:,:,i);
    Threshes(i,1) = graythresh(NormImage(:,:,i));
    OtsuMask(:,:,i) = im2bw(NormImage(:,:,i),Threshes(i,1));
end

%Create fiducial image
[FID] = FID_seg(OtsuMask);

%Make leg outline
[Leg] = Leg_seg(OtsuMask,FID);

%Segment tumor
[Tumor] = Tumor_seg(Leg,Image);
    
end
