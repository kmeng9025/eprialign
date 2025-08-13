% st_elm = epr_strel(el_type, el_sz)
% 3D structural element
%  el_type  'cube' or 'sphere'
%  el_sz    number of layers to erode

% boep, 2009

function st_elm = epr_strel(el_type, el_sz)

switch el_type
  case 'cube'
    st_elm = ones((2*el_sz+1)*[1,1,1]);
  case 'sphere'
    st_elm = zeros((2*el_sz+1)*[1,1,1]);
    dim = (1:(2*el_sz+1))';
    dim1 = dim(:, ones((2*el_sz+1), 1), ones((2*el_sz+1), 1)); %x
    dim2 = permute(dim1, [2,1,3]); %z
    dim3 = permute(dim1, [2,3,1]); %y
    
    st_elm((dim1-el_sz-1).^2+(dim2-el_sz-1).^2 + (dim3-el_sz-1).^2 <= el_sz^2) = 1;
  case 'ellipse'
    inflation = fix(el_sz(1:3));
    inflation(inflation == 0) = 1e-20;
%     inflation
    center    = max(inflation)+1;
    st_elm = zeros((2*max(inflation)+1)*[1,1,1]);
    dim = (1:(2*max(inflation)+1))';
    [dim2, dim1, dim3] = meshgrid(dim, dim, dim);
    
    st_elm(((dim1-center)/inflation(1)).^2+((dim2-center)/inflation(2)).^2 + ((dim3-center)/inflation(3)).^2 <= 1) = 1;
end