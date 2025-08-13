function res = load_aperture_data(production_folder, parameter)

% loading options
str = search_production_datasets(production_folder);

if(numel(str) == 0)
  warning('Production data set is absent.');
else
  if(numel(str) > 1)
    if isempty(parameter.ProductionFolder)
      indx = listdlg('ListString',str, 'Name', 'Production file');
      if ~isempty(indx)
        production_file = str{indx};
      end
    elseif strcmp(str{1}, 'Plug_production_dataset') == true
      production_file = str{1};
    else
      production_file = str{1};
      for ii=1:length(str)
        if strcmp(str{ii}, [parameter.ProductionFolder, '\', 'production_data.mat']) == true
          production_file = str{ii}; break;
        end
      end
    end
  else
    production_file = str{1};
  end
  
  if contains(production_file, 'Plug_production_dataset')
    fprintf('Loading experimental dataset %s\n', production_file);
    LoadedProductionFile = load(fullfile(production_folder, production_file));
    
    if isfield(LoadedProductionFile, 'CT_Frame_data') && isfield(LoadedProductionFile.CT_Frame_data, 'Tumor_CT')
      res.CTFrame.Tumor = LoadedProductionFile.CT_Frame_data.Tumor_CT;
    end
    if isfield(LoadedProductionFile, 'CT_Frame_data') && isfield(LoadedProductionFile.CT_Frame_data, 'Hypoxia')
      res.CTFrame.Hypoxia = LoadedProductionFile.CT_Frame_data.Hypoxia;
    end
    if isfield(LoadedProductionFile, 'Bev_masks')
      res.Bev_masks = LoadedProductionFile.Bev_masks;
    end
    for ii=1:length(res.Bev_masks)
      all_masks =res.Bev_masks{ii};
      if ~isfield(all_masks, 'Hypoxia'), all_masks.Hypoxia = all_masks.Boost_map; end
      if ~isfield(all_masks, 'Boost'), all_masks.Boost = all_masks.Dilated_boost_map; end
      if ~isfield(all_masks, 'Antiboost')
        if isfield(all_masks, 'Antiboost_map')
          all_masks.Antiboost = all_masks.Antiboost_map;
        else
          all_masks.Antiboost = false(size(all_masks.Boost));
        end
      end
      % project Tumor into beam plane
      beam_center = epr_maskcm(LoadedProductionFile.CT_Frame_data.Hypoxia_CT);
      beam_center = beam_center([2,1,3]);
      [all_masks.Tumor] = ...
        imrt_maskbev( res.Bev_masks{ii}.Angle, LoadedProductionFile.CT_Frame_data.Tumor_CT, beam_center, [] );
 
     res.Bev_masks{ii} = all_masks;
    end
  elseif contains(production_file, 'production_data.mat') 
    fprintf('Loading production dataset %s\n', production_file);
    LoadedProductionFile = load(fullfile(production_folder, production_file));
    if isfield(LoadedProductionFile, 'Bev_masks')
      res.Bev_masks = LoadedProductionFile.Bev_masks;
    end
    for ii=1:length(res.Bev_masks)
      all_masks = res.Bev_masks{ii};
      if ~isfield(all_masks, 'Hypoxia'), all_masks.Hypoxia = all_masks.Boost_map; end
      if ~isfield(all_masks, 'Boost'), all_masks.Boost = all_masks.Dilated_boost_map; end
      if ~isfield(all_masks, 'Antiboost')
        all_masks.Antiboost = false(size(all_masks.Boost));
      end
      res.Bev_masks{ii} = all_masks;
    end
  else
    warning('Production data set is absent.');
  end
end

fprintf('Beams are loaded:\n');
for ii=1:length(res.Bev_masks)
  all_masks = res.Bev_masks{ii};
  fprintf("  Angle %4i: Hypox=%i Boost=%i ABoost=%i \n", ...
    all_masks.Angle, ...
    numel(find(all_masks.Hypoxia)),...
    numel(find(all_masks.Boost)),...
    numel(find(all_masks.Antiboost)));
end


% --------------------------------------------------------------------
function str = search_production_datasets(fpath)
str = {};
if exist(fullfile(fpath, 'Plug_production_dataset.mat'), 'file') == 2
  str{end+1} = 'Plug_production_dataset';
end
files = dir(fpath);
dirFlags = [files.isdir];
subDirs = files(dirFlags);

for ii=3:numel(subDirs)
  if(exist(fullfile(fpath, subDirs(ii).name, 'production_data.mat'), 'file') == 2)
    str{end+1} = fullfile(subDirs(ii).name, 'production_data.mat');
  end
end
