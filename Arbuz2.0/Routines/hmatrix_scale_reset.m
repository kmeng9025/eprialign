function outmat = hmatrix_scale_reset(inmat)
%HMATRIX_SCALE_RESET reset scaling to unit

% [ Rxx Ryx Ryz 0 ]
% [ Rxx Ryx Ryz 0 ]
% [ Rxx Ryx Ryz 0 ]
% [ Tx  Ty  Tz  1 ]

outmat = inmat;
sz = size(inmat);
for ii=1:sz(2)-1, 
  a = outmat(1:sz(1)-1, ii); 
  outmat(1:sz(1)-1, ii) = a/norm(a); 
end
