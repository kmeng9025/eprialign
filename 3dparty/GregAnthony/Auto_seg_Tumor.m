%%Automated leg tumor segmentation for axial MRI images of C3H mice%%
%Greg Anthony
%2015

%Before running, run the Center MATLAB toolbox path and export the image 
%sequence to be segmented with the Arbuz GUI. 

function [Tumor,HandDrawnTumor] = Auto_seg_Tumor(export)

Image = export.Images{1}.Image;
LegMask = export.Masks{2}.Mask;
FIDMask = export.Masks{3}.Mask;
TumorMask = export.Masks{4}.Mask;
Leg = LegMask.*Image;
FID = FIDMask.*Image;
HandDrawnTumor = TumorMask.*Image;

ibGUI(Leg)

[Tumor] = Tumor_seg(LegMask,Image);

end

%%%