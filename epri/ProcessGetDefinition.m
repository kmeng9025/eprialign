function field_def = ProcessGetDefinition(definition, group, field)

field_def = []; 

for ii=1:length(definition)
  if isequal(definition{ii}.Field, group)
    p_list = definition{ii}.Parameters;
    for jj=1:length(p_list)
      if isequal(p_list{jj}.Field, field)
        field_def = p_list{jj};
        return;
      end
    end
    return;
  end
end
