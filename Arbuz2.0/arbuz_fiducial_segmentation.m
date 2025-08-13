function [outline, fiducials] = arbuz_fiducial_segmentation(raw_image, opts)
outline = false(size(raw_image));
fiducials = false(size(raw_image));
sz = size(raw_image);

first_slice = safeget(opts, 'first_slice', 1);
last_slice = safeget(opts, 'last_slice', 1000);
last_slice = min(last_slice, sz(3));

image_threshold = safeget(opts, 'image_threshold', 0.17);
largest_noise_object = safeget(opts, 'largest_noise_object', 20);
fiducials_threshold = safeget(opts, 'fiducials_threshold', 0.4);
dilate_radius = safeget(opts, 'dilate_radius', 4);
extract_outline = safeget(opts, 'extract_outline', true);
look_for_expansion = safeget(opts, 'look_for_expansion', true);
safety_mask = safeget(opts, 'safety_mask', []);

% clear image
slice_idx = false(sz(3),1);
slice_idx(first_slice:last_slice) = true;
raw_image(:,:,~slice_idx) = 0;

if ~isempty(safety_mask)
  raw_image(~safety_mask)=0;
end

% potential estimation
% thresh = multithresh(raw_image, 1);

maxim = max(raw_image(:));
mask = raw_image > image_threshold*maxim;

% image processing to close small parts away of main object
scale = size(raw_image); scale = scale / max(scale); 
se = strel_ellipsoid(dilate_radius*scale);
mask = imerode(imdilate(mask, se), se);
mask = imfill(mask,'holes');

CC = bwconncomp(mask);
PixelIdxList = zeros(CC.NumObjects, 1);
for jj=1:CC.NumObjects
  PixelIdxList(jj)=length(CC.PixelIdxList{jj});
end

% select maximum one as an outline
if extract_outline
  [~,outline_idx] = max(PixelIdxList);
  outline(CC.PixelIdxList{outline_idx}) = true;
else
  outline_idx = -1;
end

% select large remains as fiducials
for jj=1:CC.NumObjects
  if jj~=outline_idx && PixelIdxList(jj) > largest_noise_object
    if max(raw_image(CC.PixelIdxList{jj})) > fiducials_threshold*maxim
      fiducials(CC.PixelIdxList{jj}) = true;
    end
  end
end

% expand fiducials
if look_for_expansion
  se = strel_ellipsoid(1*[1,1,1]);
  new_image = raw_image;
  new_image(~imdilate(fiducials, se)) = 0;
  opt1 = opts;
  opt1.look_for_expansion = false;
  opt1.image_threshold = image_threshold / 2;
  opt1.fiducials_threshold = fiducials_threshold / 5;
  opt1.extract_outline = false;
  [~, fiducials] = arbuz_fiducial_segmentation(new_image, opt1);
end


function se=strel_ellipsoid(dilate_radius)
R = max(dilate_radius);
se = strel('sphere',R);
sz = size(se.Neighborhood);
[x,y,z]=meshgrid(1:sz(1), 1:sz(2), 1:sz(3));
Neighborhood = zeros(sz);
scale = 1./(dilate_radius/max(dilate_radius)).^2; 
Neighborhood(scale(1)*(x-sz(1)/2-0.5).^2+scale(2)*(y-sz(2)/2-0.5).^2+scale(3)*(z-sz(3)/2-0.5).^2 <= R^2) = 1;
se = strel('arbitrary',Neighborhood);
