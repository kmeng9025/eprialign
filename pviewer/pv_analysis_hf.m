function pv_analysis_hf(the_list, parameters)

threshold = safeget(parameters, 'threshold', 10);
image_name = safeget(parameters, 'image', 'PO2_pEPRI');
mask_name = upper(safeget(parameters, 'mask', 'tumor'));

data_list = {};
for ii=1:length(the_list)
  if ~isempty(the_list{ii}.registration)
    data_list{end+1} = the_list{ii};
  end
end

% disp(data_list);

processing_list = {};

hh = figure(100);

% statistics parameters
pars.hf = threshold;
pars.e  = [0 1];

fprintf('\nStandard HF processing script\n');
fprintf('threshold = %f\n', threshold);
fprintf('image = %s\n', image_name);
fprintf('mask = %s\n', mask_name);

%Changed to find all HFs for all pO2 images and display them as output
tic
for ii=1:length(data_list)
  arbuz_OpenProject(hh,data_list{ii}.registration);
  % Find all oxygen images
  search_result = arbuz_FindImage(hh, 'all','ImageType', 'PO2_pEPRI', {'SlaveList'});
  
  any_mask_found = false;
  for jj=1:length(search_result)
    mask_found = false;
    % Find all oxygen tumor masks
    search_result2 = arbuz_FindImage(hh, search_result{jj},'FINDSLAVESWITHINNAME', mask_name, {''});
    search_result2 = arbuz_FindImage(hh, search_result2,'FINDSLAVESIMAGETYPE', '3DMASK', {''});

    if ~isempty(search_result2)
      my_message = '';
      for kk=1:length(search_result2)
        if isempty(strfind(upper(search_result2{kk}.Slave), mask_name))
          continue;
        end
        any_mask_found = true;
        mask_found = true;
        processing_list{end+1} = struct('REG', data_list{ii}, 'PO2',search_result{jj},'MASK', search_result2{kk});
        processing_list{end}.STAT = my_stat(hh, processing_list{end}, pars);
        my_message = [my_message, ', ', search_result2{kk}.Slave];
      end
      if ~mask_found
        %         disp([data_list{ii}.tag, ' ', search_result{jj}.Image, ':',my_message,' -- masks_found']);
      end
    end
  end
  if ~any_mask_found
    disp([data_list{ii}.tag, ' ', search_result{jj}.Image,' warning - no mask']);
  end
end
toc
close(100);
end

function stat = my_stat(hh, rec, pars)

search_result = arbuz_FindImage(hh, rec.PO2,'', '', {'data','mask'});
the_image = search_result{1}.data;
the_po2_mask = search_result{1}.Mask;
search_result = arbuz_FindImage(hh, rec.MASK,'', '', {'data'});
MaskT = search_result{1}.data;

MaskT = MaskT & the_po2_mask;

for hf=pars.hf
  for erosion=pars.e
    NFLD = sprintf('n');
    HFFLD = sprintf('HF%i',hf);
    if erosion > 0,
      NFLD = sprintf('%sE%i',NFLD,erosion);
      HFFLD = sprintf('%sE%i',HFFLD,erosion);
      EMask = imerode(MaskT, epr_strel('sphere', erosion));
    else
      EMask = MaskT;
    end
    
    stat.(NFLD) = numel(find(EMask));
    stat.(['n',HFFLD]) = numel(find(the_image(EMask) < hf));
    stat.(HFFLD) = stat.(['n',HFFLD])/stat.(NFLD);
  end
end

disp(sprintf('%s-%s-%s: E0=%4.2f(%i) E1=%4.2f(%i)', ...
  rec.REG.tag, rec.PO2.Image, rec.MASK.Slave, stat.HF10, stat.n, ...
  stat.HF10E1, stat.nE1))
end
