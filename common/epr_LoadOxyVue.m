function [return_structure, pars] = epr_LoadOxyVue(fid_file)

fid = []; pars = []; return_structure = [];

TDMS = epr_LoadTDMS(fid_file);

script = TDMS.root.script;
json = jsondecode(script);

filetype = TDMS.root.type;

switch(filetype)
  case 'Dense_v1'
    dddd = TDMS.streams.data{1};
    dims = sscanf(TDMS.axes.dim, '%i,%i,%i,%i');
    return_structure.Raw = reshape(dddd, flip(dims'));
    return_structure.Raw = squeeze(permute(return_structure.Raw, [4,3,2,1]));
    x = TDMS.axes.data{1};
    y = TDMS.axes.data{2};
    z = TDMS.axes.data{3};
    return_structure.Size = max(x) - min(x);
    return_structure.ImageName = TDMS.instrument.name;
    
    if size(TDMS.axes.data) > 3
      t = TDMS.axes.data{4};
      return_structure.raw_info.T1 = t;
      return_structure.raw_info.data.Modality = 'PULSEFBP';
      return_structure.raw_info.data.Sequence = 'ESEInvRec';
    end
  case 'Sparse_v1'
    dims = sscanf(TDMS.axes.dim, '%i,%i,%i');
    x = TDMS.streams.data{1};
    y = TDMS.streams.data{2};
    z = TDMS.streams.data{3};
    
    mask = TDMS.streams.data{4};
    R1 = TDMS.streams.data{7};
    Amp = TDMS.streams.data{5};
    
    pO2 = (R1 - json.clb.interceptT1) * json.clb.slopeT1;
    pO2(pO2 < -25) = 25;
    pO2(pO2 > 200) = 200;
    
    return_structure.Amp = zeros(dims(:)');
    return_structure.pO2 = zeros(dims(:)');
    return_structure.Mask = false(dims(:)');
    
    idx = sub2ind(size(return_structure.Amp), x,y,z);
    return_structure.Amp(idx) = Amp;
    return_structure.pO2(idx) = pO2;
    return_structure.Mask(idx) = mask;
    
    x = TDMS.axes.data{1};
    return_structure.Size = max(x) - min(x);
end

% --------------------------------------------------------------------
function ret = get_option(val, opt_def, opt_val, opt_deflt)
for ii=1:length(opt_def)
  if(strcmp(val,opt_def{ii}))
    ret = opt_val{ii};
    return;
  end
end
ret = opt_deflt;

% --------------------------------------------------------------------
function directoryname = find_root(directoryname)
parts = strsplit(directoryname, filesep);
the_root = parts{1};
parts = parts(2:end);

for ii=length(parts):-1:0
  directoryname = the_root;
  if ii > 0
    for jj=1:ii
      directoryname =[directoryname , filesep, parts{jj}]; %#ok<AGROW>
    end
  end
  
    if exist([directoryname, filesep, 'acqp'], 'file')
      return;
    else
      directoryname = [];
    end
end

