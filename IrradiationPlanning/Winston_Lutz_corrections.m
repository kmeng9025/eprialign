function WL_shift = Winston_Lutz_corrections()

%Find the W_L values before we get into the BEVS.
%Update the table for these values, the one below was used for all of IMRT3
%IMRT 4 and IMRT 5
%Table_name = 'Z:\CenterProjects\IMRT_FSa\IMRT_3_development\Plug_Shift_table\Final_plug_shift_table_10_ports.xlsx' ;

%This table is the result of Scott Trinkle's rotation work with the group.
% Table_name = 'Z:\CenterProjects\IMRT_FSa\IMRT_3_development\Plug_Shift_table\Final_plug_shift_table_10_ports_update_05_08_17.xlsx' ;
Table_name = 'z:\CenterProjects\IMRT_FSa\IMRT_3_development\Plug_Shift_table\Final_plug_shift_table_10_ports_update_06_14_17.xlsx' ;
%Old WL table
% Table_name = 'Z:\CenterProjects\IMRT_FSa\IMRT_3_development\Plug_Shift_table\Final_plug_shift_table_10_ports_IMRT3_4_5.xlsx' ;

% BE issue with XL opening 11/16/22
% W_L_table = xlsread(Table_name);
stored = open('Z:/CenterMATLAB/Calibration/W_L_table.mat');
W_L_table = stored.W_L_table;

for ii= 1:10
% WL_shift{ii}.Plug_X = W_L_table(ii,4) + W_L_table(ii,7); 
% WL_shift{ii}.Plug_Y = W_L_table(ii,5) + W_L_table(ii,8);
Gantry_angle(ii) = W_L_table(ii,1);
Plug_X(ii) = W_L_table(ii,3);
Plug_Y(ii) = W_L_table(ii,4);
end
WL_shift = struct('Gantry_angle',Gantry_angle,'Plug_X',Plug_X,'Plug_Y',Plug_Y);
