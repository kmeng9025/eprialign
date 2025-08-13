function [Status] = epr_excel_save(FileName,CellToWrite)
%Write the Cell to the filename.xls I should have generalized this code
%long ago
%10/6/17 MM

Status = 0;

[path, fname]  = fileparts(FileName);
epr_mkdir(path);


%The hard way.
e = actxserver('Excel.Application');
e.DisplayAlerts = false; % Suppress Excel warning popups, like for overwriting a file.
eWorkbook = e.Workbooks.Add;
e.Visible = 1;
eSheets = e.ActiveWorkbook.Sheets; 
eSheet1 = eSheets.get('Item',1);
eSheet1.Activate
%get a matlab var that represents the excel range

Range = sprintf('%s','A1:',horzcat(Excel_column_letter_generator(size(CellToWrite,2)),num2str(size(CellToWrite,1))));
eActivesheetRange = get(e.Activesheet,'Range',Range);
%assign values to the matlab var that communicates to excel
eActivesheetRange.Value = CellToWrite;
eActivesheetRange.Select;
e.Selection.Columns.AutoFit; 
e.Selection.FormatConditions.Delete;


%add some conditional formatting (Not implemented)
%This works as a sample, general conditional formatting is too hard so use
%matlab flow control and buisness logic to figure and than these simple
%commands to control the font colors. Probably more work than it is worth.
% Range = sprintf('%s','A1:',horzcat(Excel_column_letter_generator( size(CellToWrite,2)),num2str(size(CellToWrite,1))));
% eWorkbook.Worksheets.Item(1).Range(Range).Interior.ColorIndex = 1;
% eWorkbook.Worksheets.Item(1).Range(Range).Font.ColorIndex = 3;

SaveAs(eWorkbook,FileName) %Save the file
eWorkbook.Saved = 1;
Close(eWorkbook) %Close the workbook
Quit(e) %Quit the program
delete(e) %Delete the server object


disp(sprintf('%s','File written to path:  ' , FileName));
% 
 
end

