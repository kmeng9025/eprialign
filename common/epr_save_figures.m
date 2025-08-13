function [  ] = epr_save_figures( Figure_cell , Title_cell, params )
%Function to save figures generated to some path (params.output/Figure_save/Figure_save_date) with titles 
%Make sure the figures are saved in a cell array, if there is one put it as
%{h}, make sure the title cell is same length as figure cell, and params has an output_path field.

%MM 2/21/2017
date = datetime('now');
Datestring = sprintf('%s',num2str(date.Month),'_',num2str(date.Day),'_',num2str(date.Year));
Timestring = sprintf('%s',num2str(date.Month),'_',num2str(date.Day),'_',num2str(date.Year),'_',num2str(date.Minute),'_',num2str(round(date.Second)));


for ii = 1:length(Figure_cell)
    Out_file = sprintf('%s',params.output_path,filesep,'Figure_save',filesep,'Figure_save_',Datestring,filesep,Title_cell{ii},'_',Timestring,'.fig');
    epr_mkdir(fileparts(Out_file));
    Figure_cell{ii}.UserData = params;
    saveas(Figure_cell{ii},Out_file);
    disp(sprintf('%s','Saved figure at path: ', Out_file ));
end

end

