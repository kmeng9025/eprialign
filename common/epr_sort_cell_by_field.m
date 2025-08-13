function [ Sorted_cell , Sort_idx ] = epr_sort_cell_by_field(Cell,field_str)
%Generalized function to take a cell array with 1xn structure variables and
%run the sort on one of the fields from the struct. Useful because most of
%our databases are this form. 

%MM 12/16/2016
    for ii = 1:length(Cell)
       sort_cell{ii} =  Cell{ii}.(field_str);
    end
      [~ , Sort_idx]  = sort(sort_cell);      
    for ii = 1:length(Cell)
       Sorted_cell{ii} =  Cell{Sort_idx(ii)};
    end
end



