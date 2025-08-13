function calculate_beam_parameters(the_list, parameters)

parameters.dataset = safeget(parameters, 'dataset', '');
parameters.max_hf = 10; % no restrictions
parameters.first_date = datenum('30-Sep-2016');
parameters.last_date  = datenum('1-Sep-2019');

processed_data_file_name = 'coverage_map.txt';
% processing_algorithm = 'calculate_beams';
processing_algorithm = safeget(parameters, 'processing_algorithm', 'load_statistics');

[boost, aboost, dbase] = pv_imrt_select(the_list, parameters);

switch processing_algorithm
  case 'calculate_beams'
    for ii=1:length(dbase)
      calculate(dbase{ii}, processed_data_file_name);
    end
  case 'load_statistics'
    for ii=1:length(dbase)
      items = [];
      fpath = fileparts(dbase{ii}.registration);
      fname = fullfile(fpath, 'PROCESSING', processed_data_file_name);
      fid = fopen(fname, 'r');
      if fid ~= -1
        while ~feof(fid)
          str = fgets(fid);
          res = regexp(str, '\s*boost:\s*(?<item1>\d+)\s*overall (?<item2>\d+) \s*tumor (?<item3>\d+) \s*hypoxia (?<item4>\d+)', 'names');
          if ~isempty(res) 
            items.boost = str2double(res.item1); 
            items.overall = str2double(res.item2); 
            items.tumor = str2double(res.item3); 
            items.hypoxia = str2double(res.item4); 
          end
        end
        fclose(fid);
      else
        fprintf('Calculating (%i) %s\n', ii, fname);
        calculate(dbase{ii}, processed_data_file_name);
        
        % try to reload after calculations
        fid = fopen(fname, 'r');
        if fid ~= -1
          while ~feof(fid)
            str = fgets(fid);
          res = regexp(str, '\s*boost:\s*(?<item1>\d+)\s*overall (?<item2>\d+) \s*tumor (?<item3>\d+) \s*hypoxia (?<item4>\d+)', 'names');
          if ~isempty(res) 
            items.boost = str2double(res.item1); 
            items.overall = str2double(res.item2); 
            items.tumor = str2double(res.item3); 
            items.hypoxia = str2double(res.item4); 
          end
          end
          fclose(fid);
        else
          fprintf('Error loading (%i) %s\n', ii, fname);
            items.boost = 0; 
            items.overall = 1; 
            items.tumor = 0; 
            items.hypoxia = 0; 
        end
      end
      dbase{ii}.stat = items;
    end
    
    boost = cellfun( @(x) (x.stat.boost), dbase);
    hypoxia = cellfun( @(x) (x.stat.hypoxia), dbase);
    tumor = cellfun( @(x) (x.stat.tumor), dbase);
    overall = cellfun( @(x) (x.stat.overall), dbase);
    
    data(:,1) = hypoxia;
    data(:,2) = tumor-hypoxia;
    data(:,3) = overall-tumor;   
    figure(5); clf
    subplot(2,1,1)
    idx = boost == 1;
    data1 = data(idx,:);
%     data1(~idx,:) = 0;
    bar(data1, 'stacked')
    legend({'hypoxia', 'normoxia','outside'})
    title(sprintf('%s: Hypoxia boost', parameters.dataset));
    subplot(2,1,2)
    idx = boost == 0;
    data1 = data(idx,:);
%     data1(~idx,:) = 0;
    bar(data1, 'stacked')
    legend({'hypoxia', 'normoxia','outside'})
    title(sprintf('%s: Normoxia boost', parameters.dataset));
    
    map = brewermap(3,'Set1');
    range = linspace(0, max(sum(data, 2)),40);
    figure(6); clf
    subplot(2,1,1)
    idx = boost == 1;
    data1 = data(idx,:);
    histf(data1(:,3),range,'facecolor',map(1,:),'alpha',.5,'edgecolor','k'); hold on
    histf(data1(:,1)+data1(:,2),range,'facecolor',map(2,:),'alpha',.5,'edgecolor','k');
    legend({'outside', 'inside'}); axis tight
    title(sprintf('%s: Hypoxia boost', parameters.dataset));
    subplot(2,1,2)
    idx = boost == 0;
    data1 = data(idx,:);
    histf(data1(:,3),range,'facecolor',map(1,:),'alpha',.5,'edgecolor','k'); hold on
    histf(data1(:,1)+data1(:,2),range,'facecolor',map(2,:),'alpha',.5,'edgecolor','k');
    legend({'outside', 'inside'}); axis tight
    title(sprintf('%s: Normoxia boost', parameters.dataset));
end

function calculate(entry, processed_data_file_name)
try
  % load registration
   registration_path = fileparts(entry.registration);
  % load blocks
  par = [];
  par.ProductionFolder = 'IMRT';
  production_data = load_aperture_data(registration_path, par);

  if entry.boost == 1
    overall_hit = numel(find(production_data.Bev_masks{1}.Boost > 0.5));
    tumor_hit = numel(find(production_data.Bev_masks{1}.Tumor&production_data.Bev_masks{1}.Boost > 0.5));
    hypoxia_hit = numel(find(production_data.Bev_masks{1}.Boost&production_data.Bev_masks{1}.Hypoxia&production_data.Bev_masks{1}.Tumor > 0.5));
  else
    overall_hit = numel(find(production_data.Bev_masks{1}.Antiboost > 0.5));
    tumor_hit = numel(find(production_data.Bev_masks{1}.Tumor&production_data.Bev_masks{1}.Antiboost > 0.5));
    hypoxia_hit = numel(find(production_data.Bev_masks{1}.Antiboost&production_data.Bev_masks{1}.Hypoxia&production_data.Bev_masks{1}.Tumor > 0.5));
  end
catch
  overall_hit = 0;
  tumor_hit = 0;
  hypoxia_hit = 0;
  disp(['calculate_coverage_map:calculate:calculation error for ',entry.registration]);
end
  fid = fopen(fullfile(registration_path, 'PROCESSING', processed_data_file_name), 'w');
  fprintf(fid, sprintf('boost: %i overall %i tumor %i hypoxia %i\n', entry.boost, overall_hit, tumor_hit, hypoxia_hit));
  fclose(fid);
  disp('Processing file has been saved.');
