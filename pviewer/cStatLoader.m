classdef cStatLoader < handle
  properties (Constant)
    SingleDose = {'subject', 3; 'days', 2; 'volume', 8; 'derm', 10; 'weight', 11; 'dose', 16};
    FracDose = {'subject', 3; 'days', 2; 'volume', 9; 'derm', 11; 'weight', 12; 'dose', 17};
  end
  properties
    Fields
    subj_array
    day_array
    RawData
    MouseData
    Source
  end
  methods
    function this = cStatLoader(new_fields)
      this.Fields = new_fields;
    end
    function Import(this, filename)
      [subj_idx, days_idx, this.RawData] = this.get_fields(this.Fields);
      [~, ~, alldata] = xlsread(filename,'Sheet1');
      
      % load subject column
      this.subj_array = this.convert(alldata(:,subj_idx), [], 'pv_mouse_stat_loader:subject');
      % ignore data after first NAN
      maxnan = find(isnan(this.subj_array), 1);
      this.subj_array = this.subj_array(1:maxnan-1);
      
      % load other columns
      this.day_array = this.convert(alldata(:,days_idx), maxnan-1, 'pv_mouse_stat_loader:day');
      for ii=1:length(this.RawData)
        this.RawData{ii}.array = this.convert(alldata(:,this.RawData{ii}.col), maxnan-1, ['pv_mouse_stat_loader:',this.RawData{ii}.field]);
      end
      
      idxs = unique(this.subj_array);
      this.MouseData = cell(max(idxs),1);
      for id = 1:max(idxs)
        for ii=1:length(this.RawData)
          ddidx = find(this.subj_array == id);
          vv = [this.day_array(ddidx), this.RawData{ii}.array(ddidx)];
          vv = vv(~isnan(vv(:,2)), :);
          this.MouseData{id}.(this.RawData{ii}.field) = vv;
        end
      end
      [this.Source.fp, this.Source.fn] = fileparts(filename);
    end
    function Save(this)
      file_name = fullfile(this.Source.fp, [this.Source.fn, '.mat']);
      save(file_name, 'this');
    end
    function res = Load(this, file_name)
      if ~isempty(file_name)
        [this.Source.fp, this.Source.fn] = fileparts(file_name);
      end
      file_name = fullfile(this.Source.fp, [this.Source.fn, '.mat']);
      res = load(file_name);
      this.MouseData = res.this.MouseData;
      this.Fields = res.this.Fields;
    end
    function SingleMouse(this, id)
      figure;
      subplot(2,2,1)
      plot(this.MouseData{id}.volume(:,1), this.MouseData{id}.volume(:,2))
      xlabel('days'); ylabel('volume'); axis tight;
      subplot(2,2,2)
      plot(this.MouseData{id}.weight(:,1), this.MouseData{id}.weight(:,2))
      xlabel('days'); ylabel('weight'); axis tight;
      subplot(2,2,3)
      plot(this.MouseData{id}.derm(:,1), this.MouseData{id}.derm(:,2))
      xlabel('days'); ylabel('derm'); axis tight;
      subplot(2,2,4)
      plot(this.MouseData{id}.dose(:,1), this.MouseData{id}.dose(:,2))
      xlabel('days'); ylabel('dose'); axis tight;
    end
    function AllMice(this, caption)
      doses = [];
      for ii = 1:length(this.MouseData)
        doses = [doses;this.MouseData{ii}.dose(:,2)];
      end
      doses = unique(doses,'sorted');
      dose_colors = linspecer(length(doses));
      dose_labels = {};
      for ii = 1:length(doses)
        dose_labels{ii}=sprintf('%i Gy',doses(ii));
      end
      
      hh = figure; clf
      subplot(2,2,1); cla; hold on
      % plot growth rate
      for ii = 1:length(this.MouseData)
        % Generate scatter plot of days vs growth rate color coded by dose,
        % previously established.
        volume = this.MouseData{ii}.volume;
        dose = mean(this.MouseData{ii}.dose(:,2));
        [~,didx] = min(abs(doses-dose));
        scatter(volume(:,1),volume(:,2),...
          'MarkerEdgeColor',dose_colors(didx,:),...
          'MarkerFaceColor',dose_colors(didx,:),...
          'MarkerEdgeAlpha',0.4,'LineWidth',1);
      end
      xlabel('Days'); ylabel('Tumor Caliper Volume [mm^3]');
      axis tight;
      set(gca,'FontSize',14)
      title([caption, ': Tumor Volume'],'FontSize',16)
      
      % Make legend
      h = zeros(length(doses), 1);
      for ii=1:length(doses)
        h(ii) = plot(NaN,NaN,'o');
        set(h(ii),'MarkerEdgeColor',dose_colors(ii,:), 'MarkerFaceColor',dose_colors(ii,:));
      end
      legend(h, dose_labels);
      %       saveas(f,'SCC7_frac_vol.png');
      
      subplot(2,2,2); cla; hold on
      % plot derm score
      for ii = 1:length(this.MouseData)
        derm = this.MouseData{ii}.derm;
        dose = mean(this.MouseData{ii}.dose(:,2));
        [~,didx] = min(abs(doses-dose));
        % Generate scatter plot of days vs growth rate color coded by dose,
        % previously established.
        scatter(derm(:,1),derm(:,2),...
          'MarkerEdgeColor',dose_colors(didx,:),...
          'MarkerFaceColor',dose_colors(didx,:),...
          'MarkerEdgeAlpha',0.4,'LineWidth',1);
      end
      xlabel('Days'); ylabel('Dermatitis Score');
      axis tight
      set(gca,'FontSize',14)
      title([caption, ': Dermatitis Score'],'FontSize',16)
      
      % Make legend
      h = zeros(length(doses), 1);
      for ii=1:length(doses)
        h(ii) = plot(NaN,NaN,'o');
        set(h(ii),'MarkerEdgeColor',dose_colors(ii,:), 'MarkerFaceColor',dose_colors(ii,:));
      end
      legend(h, dose_labels);
      %       saveas(f,'SCC7_frac_derm.png');
      
      % plot weights
      subplot(2,2,3); cla; hold on
      for ii = 1:length(this.MouseData)
        weight = this.MouseData{ii}.weight;
        dose = mean(this.MouseData{ii}.dose(:,2));
        [~,didx] = min(abs(doses-dose));
        % Generate scatter plot of days vs growth rate color coded by dose,
        % previously established.
        scatter(weight(:,1),weight(:,2),...
          'MarkerEdgeColor',dose_colors(didx,:),...
          'MarkerFaceColor',dose_colors(didx,:),...
          'MarkerEdgeAlpha',0.4,'LineWidth',1);
      end
      xlabel('Days'); ylabel('Weight [grams]');
      axis tight
      set(gca,'FontSize',14)
      title([caption, ': Weight'],'FontSize',16)
      
      % Make legend
      h = zeros(length(doses), 1);
      for ii=1:length(doses)
        h(ii) = plot(NaN,NaN,'o');
        set(h(ii),'MarkerEdgeColor',dose_colors(ii,:), 'MarkerFaceColor',dose_colors(ii,:));
      end
      legend(h, dose_labels,'Location','southeast');
      %       saveas(f,'SCC7_frac_weight.png');
    end
  end
  methods (Static)
    function [subject, days, out_fields] = get_fields(in_fields)
      out_fields = [];
      subject = [];
      days = [];
      for ii=1:size(in_fields, 1)
        if strcmp(in_fields{ii,1},'subject'), subject = in_fields{ii,2};
        elseif strcmp(in_fields{ii,1},'days'), days = in_fields{ii,2};
        else
          out_fields{end+1}.field = in_fields{ii,1};
          out_fields{end}.col   = in_fields{ii,2};
        end
      end
    end
    function [rdata, header] = convert(data, max_row, estring)
      try
        header = data{1};
        
        iserror = false;
        data_class = class(data{2});
        % type check
        for ii=2:max_row
          if ~strcmp(class(data{ii}), data_class)
            iserror = true;
            fprintf("error %s:idx%i = '%s'\n", estring, ii, data{ii})
          end
        end
        
        rdata = cell2mat(data(2:end));
        
        if ~isempty(max_row)
          rdata = rdata(1:max_row);
        end
      catch err
        rethrow(err);
      end
    end
  end
end
