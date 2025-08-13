function [dice_out] = compare_dice(vol1,vol2)
%% This function calculates the Dice coefficient between two volumes.
%  The output is between 0 and 1 to determine the similarity
%    between two masks. A value of 0 indicates no overlap. A value
%    of 1 indicates totally perfect overlap.
%
%  Guidelines for inputs:
%  * vol1 = variable name of your first mask 3D volume.
%  * vol2 = variable name of your second mask 3D volume.
%
%
%  EXAMPLE: 
%  * INPUT: get_dice(tumorvol_1,tumorvol_2)
%  * OUTPUT: 0.7843 
% 
%  Questions? Email Inna Gertsenshteyn <innag@uchicago.edu>

if size(vol1) ~= size(vol2)
   error('Dimensions of your two volumes must be the same, with volumes registered to the same space!')
end
dice_out = 2 * numel(find(vol1&vol2)) / (numel(find(vol1)) + numel(find(vol2)));

return;


%  Initialize count
count1 = 0;
count2 = 0;
count3 = 0;
%  determine dimension of volumes
dim = size(vol1);
dim2 = size(vol2);
if dim ~= dim2
   error('Dimensions of your two volumes must be the same, with volumes registered to the same space!')
end


for i = 1:dim(1)
    for j = 1:dim(2)
        for k = 1:dim(3)
            if (vol1(i,j,k) == 1)
                count1 = count1 + 1;
            end
            if (vol2(i,j,k) == 1)
                count2 = count2 + 1;
            end
            if (vol1(i,j,k) == 1) && (vol2(i,j,k) == 1)
                count3 = count3 + 1;
            end
        end
    end
end

dice_out = (2 * count3) / (count1 + count2);
end