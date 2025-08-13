
function out = Pishft1(in,dim);
%

if (nargin < 2) dim =1; end

out=in;
switch ndims(in)
case 1
out(2:2:size(in,1)) = -in(2:2:size(in,1));

case 2
   switch dim
       case 1
          out(2:2:size(in,1),:) = -in(2:2:size(in,1),:);
       case 2
          out(:,2:2:size(in,2)) = -in(:,2:2:size(in,2));
       otherwise
          error ('Invalid input');
   end

case 3
   switch dim
       case 1
          out(2:2:size(in,1),:,:) = -in(2:2:size(in,1),:,:);
       case 2
          out(:,2:2:size(in,2),:) = -in(:,2:2:size(in,2),:);
       case 3
          out(:,:,2:2:size(in,3)) = -in(:,:,2:2:size(in,3));
       otherwise
          error ('Invalid input');
   end
   
case 4
   switch dim
       case 1
          out(2:2:size(in,1),:,:) = -in(2:2:size(in,1),:,:);
       case 2
          out(:,2:2:size(in,2),:) = -in(:,2:2:size(in,2),:);
       case 3
          out(:,:,2:2:size(in,3)) = -in(:,:,2:2:size(in,3));
       case 4
          out(:,:,:,2:2:size(in,4)) = -in(:,:,:,2:2:size(in,4));
       otherwise
          error ('Invalid input');
   end
   
otherwise
   error ('Invalid data set');
end


