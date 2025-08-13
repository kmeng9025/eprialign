function [out_mask, mat_fit_info] = epri_GenerateFittingMask(mat_recFXD, mat_fit_info)

switch safeget(mat_fit_info, 'fit_mask', 'threshold_mask')
  case 'threshold_mask'
    mat_fit_info.threshold = safeget(mat_fit_info, 'fit_mask_threshold', 0.1);
    mat_fit_info.fit_mask_fill = safeget(mat_fit_info, 'fit_mask_fill', 1);
    mat_fit_info.fit_mask_object_size = safeget(mat_fit_info, 'fit_mask_object_size', 0);
    mat_fit_info.data = safeget(mat_fit_info, 'data', 'decay');
    
    switch mat_fit_info.data
      case {'decay', 'recovery'}, amp_data = squeeze(max(mat_recFXD, [], 4));
      case 'absorption_line', amp_data =  squeeze(sum(mat_recFXD, 4));
    end
    
    sz = size(amp_data);
    
    % spheric mask to eliminate corners
    dim = (1:sz(1))';
    dim1 = dim(:, ones(sz(1), 1), ones(sz(1), 1)); %x
    dim2 = permute(dim1, [2,1,3]); %z
    dim3 = permute(dim1, [2,3,1]); %y
    
    r_center=(sz(1)+1)/2;
    mask1=false([sz(1),sz(2),sz(3)]);
    mask1( (dim1-r_center).^2+(dim2-r_center).^2+(dim3-r_center).^2 < (sz(1)/2-2)^2 )=true;

    mmax = max(amp_data(mask1(:)));
    cut = mmax * mat_fit_info.threshold;
    out_mask = amp_data > cut;
    adaptive_factor = 1;
    
    for ii=1:4
      stage = ['mask_proc_stage', num2str(ii)];
      switch safeget(mat_fit_info, stage, 'proc_none')
        case 'proc_adaptive_threshold'
          max_voxel = safeget(mat_fit_info, 'fit_mask_max_voxels', 10000);
          n = numel(find(amp_data(out_mask(:)) > cut));
          while n > max_voxel
            adaptive_factor = adaptive_factor * 1.05;
            cut = adaptive_factor * cut;
            n = numel(find(amp_data(out_mask(:)) > cut));
            fprintf('Adaptive threshold: %5.3f (%i).\n', adaptive_factor*mat_fit_info.threshold, n);
          end
          out_mask = amp_data > cut;
          fprintf('Adaptive threshold: %5.3f (was %5.3f).\n', adaptive_factor*mat_fit_info.threshold, mat_fit_info.threshold);
        case 'proc_leave_large'
          if mat_fit_info.fit_mask_object_size > 0
            out_mask = bwareaopen(out_mask, mat_fit_info.fit_mask_object_size, 6);
          end
        case 'proc_leave_the_largest'
            % this will leave the largest object
            CC = bwconncomp(out_mask,6);
            numPixels = cellfun(@numel,CC.PixelIdxList);
            [~,idx] = max(numPixels);
            out_mask = false(size(out_mask));
            out_mask(CC.PixelIdxList{idx}) = 1;
            fprintf('Leave the largest: %i.\n', numel(find(out_mask)));
        case 'proc_fill'
          if mat_fit_info.fit_mask_fill > 0
            out_mask = apply_imfill(out_mask, mat_fit_info.fit_mask_fill);
          end
        case 'geometric_expansion'
            mmax = max(amp_data(out_mask(:)));
            cut = mmax * mat_fit_info.threshold;

            % this will try to inflate the mask
            new_mask = imdilate(out_mask, epr_strel('sphere', 2));
            addition = new_mask & ~out_mask;
            addition(amp_data < cut) = false;
            out_mask = addition | new_mask;
            fprintf('Geometric expansion: %i (%i).\n', numel(find(addition)), numel(find(out_mask)));
      end
    end
  case 'external_file'
    ext_mask_fname = safeget(mat_fit_info, 'fit_mask_file', '');
    if exist(ext_mask_fname, 'file')
      s1 = load(ext_mask_fname);
      if isfield(s1, 'Mask')
        disp('Using external mask.');
        out_mask = s1.Mask;
        mat_fit_info.ext_mask = ext_mask_fname;
        return;
      else
        mat_fit_info.fit_mask = 'threshold_mask';
        out_mask = s1.data;
      end
    else
      mat_fit_info.fit_mask = 'threshold_mask';
      error('Mask file does not exist.');
    end
end

% --------------------------------------------------------------------
function mask = apply_imfill(mask, fill_par)
fill_par = fix(fill_par);
if fill_par < 1 || fill_par > 4, fill_par = 1; end

% figure(100)
for n=1:size(mask, 1)
  mask1=squeeze(mask(:,n,:));

  mask2 = bwmorph(mask1, 'close');
  fill_mask=~imfill(mask2, [1 1]*fill_par);
  
%   [perimBW,L,N] = bwboundaries(mask1,4,'noholes');
%   for b =1:N
%     boundary = perimBW{b};
%     if numel(boundary)>9  %island will be 2x2
%       if b==1
%         x=boundary(:,2);
%         y=boundary(:,1);
%       else  %concatenate boundaries
%         x = [x; boundary(:,2)];
%         y = [y; boundary(:,1)];
%       end
%     end
%   end

  mask(:,n,:) = mask2 | fill_mask;
%   imagesc(mask1 + 2 * xor(squeeze(mask(:,n,:)),mask1), [0,3]); colorbar;axis image
%   pause(0.3)
end
