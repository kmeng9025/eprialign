function [fout, out_pars, slave_images] = arbuz_custom_loader_cest(fname)
slave_images = {};
allvar = load(fname);

if ~isfield(allvar, 'CESTscale')
  fprintf('\n----------------------------------------------');
  error('Run CESTGUI to establish CEST image parameters');
end

fout = flip(allvar.image_r, 1);
slave_images{end+1}.data = flip(allvar.pH, 1);
slave_images{end}.A = eye(4);
slave_images{end}.Name = 'pH';

scale  = allvar.CESTscale;
extent = scale.*(size(fout)-1);
CESToffset = allvar.CESToffset;
RAREoffset = allvar.RAREoffset;
offset = [0, 0, CESToffset(1) + extent(3)/2 ]; % 
fprintf('CEST %f RARE %f\n', CESToffset(1), RAREoffset(1));

out_pars.Bbox = size(fout);
out_pars.Anative = hmatrix_translate(-(out_pars.Bbox(1:3) + 1) / 2)* ...
  hmatrix_scale(scale) * hmatrix_translate(offset);

% slave_images{1}.data = allvar.pH;
% slave_images{1}.Name = 'pH';
% slave_images{1}.A = hmatrix_scale([0.25,0.25,1]);
% slave_images{1}.isStore = 1;
% slave_images{1}.ImageType = 'CEST_ML';