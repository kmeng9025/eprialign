% --------------------------------------------------------------------
function Amp = epr_LoadSpatialImage(mat_recFXD)

sz = size(mat_recFXD);

if length(sz) < 3, Amp = [];
elseif length(sz) == 3, Amp = mat_recFXD;
elseif length(sz) == 4
  str = {}; for ii=1:sz(4), str{end+1} = sprintf('Slice %i', ii); end
  sel = listdlg('Name', 'Select the slice', 'ListString', str);
  Amp = squeeze(mat_recFXD(:,:,:,sel));
end