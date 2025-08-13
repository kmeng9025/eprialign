%Tumor Segmentation
%Greg Anthony
%2015

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[Tumor] = Tumor_seg(Leg,Image)

Tumor = Leg.*Image;

%Thresholding Segmentation
Prompt = 'Please enter a starting axial slice for the tumor...';
Start = input(Prompt);
Prompt = 'Please enter an ending axial slice for the tumor...';
End = input(Prompt);

Tumor(:,:,1:(Start-1)) = 0;
Tumor(:,:,(End+1):size(Tumor,3)) = 0;
MorphOut = 0;

while MorphOut == 0;
    ThreshOut = WindowLevelGUI(Tumor,Start,End,Image);
    
    MorphOut = MorphGUI(ThreshOut,Start,End,Image);
end    

MorphOut(MorphOut~=0)=1;

Tumor = MorphOut;
end

%%%