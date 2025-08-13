function  [animal, cast, mousebed,fiducials,bone] = ...
  arbuz_segment_CT(data, opts)

restricted_data = data;
sz = size(restricted_data);
mid_slice  = fix(sz(3)/2);
[X,Y] = meshgrid(1:sz(2),1:sz(1));
animal = false(sz);
fiducials = false(sz);
mousebed = false(sz);
cast  = false(sz);

fiducials_number = safeget(opts, 'fiducials_number', 4);
fiducials_voxels = safeget(opts, 'fiducials_voxels', 200);

first_slice = safeget(opts, 'first_slice', 1);
last_slice = safeget(opts, 'last_slice', 1000);
last_slice = min(last_slice, sz(3)-2);

cast_density_min = safeget(opts, 'cast_density_min', 2900);
cast_density_max = safeget(opts, 'cast_density_max', 4700);

animal_density_min = safeget(opts, 'animal_density_min', 400);
animal_density_max = safeget(opts, 'animal_density_max', 2000);

bed_density_min = safeget(opts, 'bed_density_min', 400);
bed_density_max = safeget(opts, 'bed_density_max', 1200);

noise_density_max = safeget(opts, 'noise_density_max', 450);

fid_density_min = safeget(opts, 'fid_density_min', 2400);
fid_density_max = safeget(opts, 'fid_density_max', 9000);

%% selections
slice_idx = false(sz(3),1);
slice_idx(first_slice:last_slice) = true;
restricted_data(:,:,~slice_idx) = 0;

% carbon bed
restricted_data(320:end,:,:) = 0;

% cast
selection = restricted_data > cast_density_min & restricted_data < cast_density_max;
se=strel('sphere', 2);
CC = bwconncomp(imdilate(imerode(selection,se),se));
items = cellfun(@(x) numel(x), CC.PixelIdxList);
[~,maxidx] = max(items);
cast(CC.PixelIdxList{maxidx}) = true;
se=strel('sphere', 4);
cast = imerode(imdilate(cast,se),se);
se=strel('sphere', 2);
suppress_cast = imdilate(cast,se);

% mouse and bed
selection = restricted_data > animal_density_min & restricted_data < animal_density_max & ~suppress_cast;
se=strel('sphere', 2);
selection = imdilate(imerode(selection,se),se);
selection(:,:,~slice_idx) = false;

CC = bwconncomp(selection);
items = cellfun(@(x) numel(x), CC.PixelIdxList);
[~,sortidx] = sort(items);
idx1 = sortidx(end);
idx2 = sortidx(end-1);

animal(CC.PixelIdxList{idx1}) = true;
mousebed(CC.PixelIdxList{idx2}) = true;
object1 = animal(:,:,mid_slice);
object1 = imfill(object1,'holes');
object2 = mousebed(:,:,mid_slice);
object2 = imfill(object2,'holes');
CM1=fix([mean(X(object1)), mean(Y(object1))]);
CM2=fix([mean(X(object2)), mean(Y(object2))]);
if ~isnan(CM2) & object2(CM2(1),CM2(2))
  a1 = animal; animal = mousebed; mousebed=a1;
end

se=strel('sphere', 8);
animal = animal | (imerode(imdilate(animal,se),se) & ~suppress_cast);
for ii=1:sz(3)
  animal(:,:,ii) = imfill(animal(:,:,ii), 'holes');
end
disp('arbuz_segment_CT: Animal outline is segmented.');

se=strel('sphere', 1);
mousebed = imdilate(mousebed,se);
disp('arbuz_segment_CT: Animal bed is segmented.');

% fiducials
selection = restricted_data > fid_density_min & restricted_data < fid_density_max & ~suppress_cast & ~animal & ~mousebed;
for ii=find(slice_idx)'
   if numel(find(selection(:,:,ii))) > fiducials_voxels*fiducials_number*2
     selection(:,:,ii) = false;
   end
end

se=strel('sphere', 1);
CC = bwconncomp(imerode(imdilate(selection,se),se));
items = cellfun(@(x) numel(x), CC.PixelIdxList);
for ii=1:length(items)
  if items(ii) > fiducials_voxels*30
    fiducials(CC.PixelIdxList{ii}) = 1;
  end
end

% remove bone
fil_fid = false(size(fiducials));
for ii=find(slice_idx)'
  fil_fid(:,:,ii) = imfill(fiducials(:,:,ii), 'holes');
end
CC = bwconncomp(fil_fid);
items = cellfun(@(x) numel(x), CC.PixelIdxList);
[~, fidx] = sort(items,'descend');
for ii=fidx(fiducials_number+1:end)
  fiducials(CC.PixelIdxList{ii}) = false;
  animal(CC.PixelIdxList{ii}) = true;
end
bone = false(sz);
[~,maxidx] = max(items);
if ~isempty(maxidx)
  bone(CC.PixelIdxList{maxidx}) = true;
end

% take only largest component
CC = bwconncomp(animal);
items = cellfun(@(x) numel(x), CC.PixelIdxList);
animal = false(sz);
[~,maxidx] = max(items);
animal(CC.PixelIdxList{maxidx}) = true;

disp('arbuz_segment_CT: Fiducials are segmented.');

%% report figure
figN  = safeget(opts, 'figure', 1000);
fname = safeget(opts, 'figure_filename', '');

if figN > -1 && ~isempty(fname)
  
  not_assigned = restricted_data > noise_density_max & ~animal ...
    & ~fiducials & ~mousebed & ~cast;
  presentation = animal + 2.5*fiducials+...
    3*mousebed+4*cast+...
    5*not_assigned;
  slice_range = find(slice_idx);
  fig_opts.legend = 'blue: animal; green: fiducials; yellow:mouse bed; orange:cast; red: not assigned';
  fig_opts.show_min = 100;
  fig_opts.show_max = 6500;
  figN = imrt_show_segmentation(figN, data, presentation, slice_range, fig_opts);
  set(figN, 'Position', get(0, 'Screensize'));
  epr_mkdir(fileparts([fname,'1.png']));
  saveas(figN, [fname,'1.png']);
  delete(figN);
end
