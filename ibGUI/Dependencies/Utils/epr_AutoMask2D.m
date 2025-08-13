% seed_mask = epr_AutoMask2D(mask_type, original_image, seed_mask, threshold)
% mask_type  'FloodFillAdjusted' or 'FloodFill' or 'Largest'

function seed_mask = epr_AutoMask2D(mask_type, original_image, seed_mask, threshold)
se = strel('disk', 1);
original_image = cast(original_image, 'double');

switch mask_type
  case 'FloodFillAdjusted',
    if numel(find(seed_mask)) < 10, seed_mask = imdilate(seed_mask, se); end
    n_it = 1;
    n_el_found_old = numel(find(seed_mask));
    dispersion = std(original_image(seed_mask(:)));
    while n_it < 20
      target_level = mean(original_image(seed_mask(:)));
      thresholds = target_level + abs(threshold * dispersion) * [-1 1];
      new_seed_mask = imdilate(seed_mask, se) ;
      new_seed_mask(original_image < thresholds(1) | original_image > thresholds(2)) = false;
      dispersion = std(original_image(new_seed_mask(:)));
      n_it = n_it + 1;
      n_el_found = numel(find(new_seed_mask));
      if n_el_found == n_el_found_old, break; end
%       if new_dispersion > dispersion*1.25, break; end
      n_el_found_old = n_el_found;
%       dispersion = new_dispersion;
      seed_mask = new_seed_mask;
%       figure(333); clf; imagesc(original_image); hold on; contour(seed_mask, 0.5,'g'); axis image
%       text(10, 160, num2str(numel(find(seed_mask))), 'Color', 'r');
%       text(70, 160, num2str(std(original_image(seed_mask(:)))), 'Color', 'r');
%       pause(.5)
    end
  case 'FloodFill',
    if numel(find(seed_mask)) < 10, seed_mask = imdilate(seed_mask, se); end
    n_it = 1;
    n_el_found_old = numel(find(seed_mask));
    thresholds = [min(original_image(seed_mask(:))), mean(original_image(seed_mask(:)))];
    while n_it < 20
      new_seed_mask = imdilate(seed_mask, se) ;
      new_seed_mask(original_image < thresholds(1) | original_image > thresholds(2)) = false;
      n_it = n_it + 1;
      n_el_found = numel(find(new_seed_mask));
      if n_el_found == n_el_found_old, break; end
      n_el_found_old = n_el_found;
      seed_mask = new_seed_mask;
%       figure(333); clf; imagesc(original_image); hold on; contour(seed_mask, 0.5,'g'); axis image
%       text(10, 160, num2str(numel(find(seed_mask))), 'Color', 'r');
%       text(70, 160, num2str(std(original_image(seed_mask(:)))), 'Color', 'r');
%       pause(.5)
    end
  case 'Largest'
    if isempty(seed_mask), return; end
    [labeled_mask, nMask] = bwlabel(seed_mask);
    mask_square = zeros(nMask, 1);
    for ii=1:nMask
      mask_square(ii) = numel(find(labeled_mask == ii));
    end
    if isempty(mask_square), return; end
    [mm, max_idx] = max(mask_square);
    seed_mask = labeled_mask == max_idx;
end
