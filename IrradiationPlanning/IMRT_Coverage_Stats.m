function [Stats] = IMRT_Coverage_Stats(BeamStruct, BeamString, VolumeData , VolumeStrings )

%Get information on what the beams hit If use the strings to access the
%right part of the struct.
Stats = [];
%Get the 3D coverage map
Coverage_map = BeamStruct{1}.(BeamString)*0;
for ii = 1:length(BeamStruct)
Coverage_map = Coverage_map + BeamStruct{ii}.(BeamString);
end

%Get the logical masks from volume data

for ii = 1:length(VolumeStrings)
   Vol =  VolumeData.(VolumeStrings{ii});
   Stats.(sprintf('%s',VolumeStrings{ii},'_Coverage')) = numel(find(Coverage_map==length(BeamStruct) & Vol== 1))/numel(find(Vol==1));
   %ibGUI(Vol - logical(Coverage_map))
end


end

