function [list, unmatched] =  epr_combine_lists (list1,list2,field_to_match, add_string)
%Takes each structure in list 1, looks at field to match and compares it to
%each structure in list 2. If a match is found, a new field in list_1 is
%created and the matched strucutre of list2 is added to the new list as a field.

%This can be made more elegant when I have time. It basically means that
%you will be matching based on a string (probably experiment name and
%creating nesting structures to hold information. Long term that is
%probably a bad idea, but merging structures that may or may not have the
%same fields is a hard problem.

%MM 12/19/2016


list = {}; unmatched = {};
for ii = 1:length(list1)
    list1{ii}.Matched = 0;
    for jj = 1:length(list2)
       if  sum(strfind(list1{ii}.(field_to_match), list2{jj}.(field_to_match)))>0
           disp(sprintf('%s','Found match for  ' , list1{ii}.(field_to_match)));
           list{end+1} = list1{ii};
           list{end}.(add_string) = list2{jj};
           list1{ii}.Matched = 1;
       end  
    end
    if ~list1{ii}.Matched
        disp(sprintf('%s','NO match found for  ' , list1{ii}.(field_to_match)));
        unmatched{end+1} = list1{ii};
    end
end


end
