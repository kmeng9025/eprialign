% OUT = search_for_data('eprima250',{'*.img','*.mat'},'C:/Temp/output_eprima250.xlsx');
% OUT = search_for_data('imagnet',{'*.img','*.mat'},'C:/Temp/output_imagnet.xlsx');
function OUT = search_for_data(root_path, extension, excel_output)

if ~iscell(root_path)
  if contains(root_path, 'eprima250')
    root_path = {'v:\data\eprima250_data_04\', 'v:\data\eprima250_data_05\', 'v:\data\eprima250_data_06\', 'v:\data\eprima250_data_07\', ...
                  'v:\data\eprima250_data_08\', 'v:\data\eprima250_data_09\', 'v:\data\eprima250_data_10\', 'v:\data\eprima250_data_11\', ...
                  'v:\data\eprima250_data_12\', 'v:\data\eprima250_data_13\', 'v:\data\eprima250_data_14\', 'v:\data\eprima250_data_15\', ...
                  'v:\data\eprima250_data_16\', 'v:\data\eprima250_data_17\'};
  end
  if contains(root_path, 'imagnet')
    root_path = {'v:\data\Imagnet_data_05\', 'v:\data\Imagnet_data_06\', 'v:\data\Imagnet_data_07\', 'v:\data\Imagnet_data_08\', ...
                  'v:\data\Imagnet_data_09\', 'v:\data\Imagnet_data_10\', 'v:\data\Imagnet_data_11\', 'v:\data\Imagnet_data_012\', ...
                  'v:\data\Imagnet_data_13\', 'v:\data\Imagnet_data_14\', 'v:\data\Imagnet_data_15\'};
  end
end

for ii=1:length(root_path)
  OUT = recursive_search(root_path{ii}, extension, {});
  sheet_name = root_path{ii};
  sheet_name(sheet_name == '\') = '.';
  sheet_name(sheet_name == '/') = '.';
  sheet_name(sheet_name == ':') = '.';
  writecell(OUT,excel_output,'Sheet',sheet_name);
end

disp('Finished !!!');
function OUT = recursive_search(root_path, extension, OUT)

files  = dir(root_path);
dirFlags = [files.isdir];
subDirs = files(dirFlags);
subDirsNames = {subDirs(3:end).name};
for ii=1:length(subDirsNames)
  OUT = recursive_search(fullfile(root_path, subDirsNames{ii}), extension, OUT);
end

for jj=1:length(extension)
  B = dir(fullfile(root_path, extension{jj}));
  for ii=1:length(B)
    OUT{end+1, 1} = B(ii).date;
    OUT{end, 2} = B(ii).name;
    OUT{end, 3} = fullfile(root_path, B(ii).name);
    OUT{end, 4} = '';
    if contains(extension{jj}, 'mat')
      try
        W = load(OUT{end, 3}, 'name_com');
        OUT{end, 4} = W.name_com;
      catch
      end
    end
  end
end

