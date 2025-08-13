function [varargout] = ProcessLoadScenario(scenario_file, parameter_file)

% test 
% epr_LoadScenario('c:\MATLAB\toolbox\time_domain\PulseRecon.scn', 'c:\MATLAB\toolbox\time_domain\PulseSOP Image Reconstruction prior to 2009-07-01.par')

if ~exist(scenario_file, 'file')
  error(sprintf('Scenario %s was not found.', scenario_file));
end
get_input_fields = @(n,x,y,t) struct('Name', n, 'Field',x,'Value',y, 'Type', t,'AutoSave',true,'Options','','Flags',0);
get_group = @(x,y,z,t) struct('Name',x,'Field',y);
groups = {};

the_line = '';
fid = fopen(scenario_file); idx = 1;
while ~feof(fid)
  the_line = fgetl(fid);
  try
    eval(the_line);
  catch err
    disp(['---', the_line, '---']);
    disp(sprintf('Error %i: %s', idx, err.message))
    idx = idx + 1;
  end
end
fclose(fid);

if exist('scenario', 'var'),  cfg.Scenario = scenario; else cfg.Scenario = fname; end

request_fld = {};
info.GroupNames = {};
% set groups popup menu
par_groups = cell(1,length(groups));
for ii=1:length(groups)
  info.GroupNames{ii} = groups{ii}.Name;
  par_groups{ii} = groups{ii};
  field_name = ['fields_', upper(groups{ii}.Field)];
  if exist(field_name, 'var')
    par_groups{ii}.Parameters = eval(field_name);
    for jj=1:length(par_groups{ii}.Parameters)
      if bitand(par_groups{ii}.Parameters{jj}.Flags,hex2dec('1'))
        request_fld{end+1}.Group = par_groups{ii}.Field;
        request_fld{end}.Field = par_groups{ii}.Parameters{jj}.Field;
      end
      par_groups{ii}.Parameters{jj}.Group = par_groups{ii}.Field;
    end
  else
    par_groups{ii}.Parameters = {};
  end
end

fprintf('Scenario is loaded from %s.\n', scenario_file);
if exist('scenario', 'var'),  info.Scenario = scenario; else info.Scenario = fname; end
if ~exist('parameter_file', 'var') || isempty(parameter_file)
  fprintf('Parameter file %s was not found.\n', scenario_file);
  info.request_fld = request_fld;
  varargout{1} = par_groups;
  varargout{2} = info;
  return;
end

ini_pars = inimanage(parameter_file);

out_fields = [];
for gr=1:length(par_groups)
  par_dsc = par_groups{gr}.Parameters;
  gr_Field = par_groups{gr}.Field;
  if isfield(ini_pars, gr_Field)
    [out_fields.(gr_Field)] = ProcessFormController(ini_pars.(gr_Field), [], par_dsc, 'ini2fields');
  else
    out_fields.(gr_Field) = [];
  end
end
disp(sprintf('Parameters are loaded from %s.', parameter_file));

varargout{1} = out_fields;
varargout{2} = par_groups;
varargout{3} = ini_pars;
