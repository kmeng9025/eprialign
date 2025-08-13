function [ Output ] = Excel_column_letter_generator( Input )

Alphabet_vector = { 'A' 'B' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N' 'O' 'P' 'Q' 'R' 'S' 'T' 'U' 'V' 'W' 'X' 'Y' 'Z' };

Len_alpha = length(Alphabet_vector);

 n = Input;
 i = 1;
 Output = '';
 string_cell = {};
 
 while n > 0
     rem = mod(n,26);
     
     if rem == 0
        string_cell{i} = Alphabet_vector{end};
        i = i + 1;
        n = (n/26)-1;     
     
     elseif rem == 1;
        string_cell{i} = Alphabet_vector{1};
        i = i + 1;
        n = floor(n/26);     
         
     else
        string_cell{i} = Alphabet_vector{rem};        
        i = i +1;
        n = floor(n/26);
        
     end
 end
 
 %Reverse the string 
for ii = length(string_cell):-1:1
   Output = sprintf('%s',Output,string_cell{ii});
end




end

