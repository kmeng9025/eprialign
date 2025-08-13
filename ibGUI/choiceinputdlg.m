function ret = choiceinputdlg(title, inputs)
%CHOICEINPUTDLG enchanced version of inputdlg
% title - Dialog title
% inputs - structure array of parameters
% inputs().type - 0-Number/1-String/2-Choices/3-Choices by index  
% inputs().name - field name
% inputs().choices = list of choices;
% inputs().default = default item for types 2 and 3
% See also INPUTDLG

height = 90+50*length(inputs);
width  = 250;
d = dialog('Position',[300 300 width height],'Name',title);

for ii=1:length(inputs)
  pos = [10,height-50*ii,width-20,25];
  inputs(3).text = uicontrol('Parent',d,...
    'Position',pos+[0,+20,-50,0],...
    'Style','text', 'HorizontalAlignment','left',...
    'String',inputs(ii).name);
  switch inputs(ii).type
    case 0
      inputs(ii).edit = uicontrol('Parent',d,...
        'Position',pos,...
        'Style','edit', ...
        'String',num2str(inputs(ii).choices));
    case 1
      inputs(ii).edit = uicontrol('Parent',d,...
        'Position',pos,...
        'Style','edit', ...
        'String',inputs(ii).choices);
    case 2
      startvalue = 1;
      if isfield(inputs(ii), 'default') && ~isempty(inputs(ii).default)
        startvalue = inputs(ii).default;
      end
      inputs(ii).edit = uicontrol('Parent',d,...
        'Position',pos,...
        'Style','popup', ...
        'String',inputs(ii).choices, ...
        'Value', startvalue);
    case 3
      startvalue = 1;
      if isfield(inputs(ii), 'default') && ~isempty(inputs(ii).default)
        startvalue = inputs(ii).default;
      end
      inputs(ii).edit = uicontrol('Parent',d,...
        'Position',pos,...
        'Style','popup', ...
        'String',inputs(ii).choices, ...
        'Value', startvalue);
  end
end

btnOk = uicontrol('Parent',d,...
  'Position',[35 20 70 25],...
  'String','Ok',...
  'Callback',@endofstory_callback);

btnCancel = uicontrol('Parent',d,...
  'Position',[155 20 70 25],...
  'String','Cancel',...
  'Callback','delete(gcf)');

ret = [];
uiwait(d);

  function endofstory_callback(popup,event)
    for jj=1:length(inputs)
      switch inputs(jj).type
        case 0
          ret{jj} = str2double(inputs(jj).edit.String);
        case 1
          ret{jj} = inputs(jj).edit.String;
        case 2
          ret{jj} = inputs(jj).edit.String{inputs(jj).edit.Value};
        case 3
          ret{jj} = inputs(jj).edit.Value;
        otherwise
          ret{jj} = '';
      end
    end
    
    delete(gcf);
  end
end