
function [return_structure, pars] = epr_LoadBrukerMRI(fid_file, read_options)

fid = []; pars = []; return_structure = [];

the_path = fileparts(fid_file);
the_root_path = find_root(the_path);

if exist('read_options','var')
else
  read_options = 'processed';
end

switch(read_options)
  case 'processed'
    reco_file = fullfile(the_path,'reco');
    proc_file = fullfile(the_path,'2dseq');
    d3proc_file = fullfile(the_path,'d3proc');
    acqp_file = fullfile(the_root_path,'acqp');
    method_file = fullfile(the_root_path,'method');
    visu_pars_file = fullfile(the_root_path,'visu_pars');
    pars.reco = epr_ReadBrukerAcqp(reco_file);
%     pars.d3proc = epr_ReadBrukerAcqp(d3proc_file);
    pars.acqp = epr_ReadBrukerAcqp(acqp_file);
    pars.method = epr_ReadBrukerAcqp(method_file);
    pars.visu_pars = epr_ReadBrukerAcqp(visu_pars_file);
    
    opt_dtype = {'_32BIT_SGN_INT', '_16BIT_SGN_INT', '_8BIT_UNSGN_INT', '_32BIT_FLOAT'};
    opt_val_dtype = {'int32', 'int16', 'uint8', 'float32'};
    dtype = get_option(pars.reco.RECO_wordtype, opt_dtype, opt_val_dtype, '?');
    
    opt_dtype = {'littleEndian','bigEndian'};
    opt_val_dtype = {'l', 'b'};
    byteorder = get_option(pars.reco.RECO_byte_order, opt_dtype, opt_val_dtype, 'l');
    
    fp=fopen(fid_file, 'r', byteorder);
    if (fp == -1)
      disp(['mrigMRI loader error: couldn''t open fid file "' fid_file '".']);
      return
    end
    fid = fread(fp, dtype);
    fclose(fp);
    
    formation = 1;
    switch formation
      case 1
        nslices = pars.acqp.NI;
        dim4 = length(safeget(pars.acqp, 'ACQ_time_points', []));
        expected_data = pars.reco.RECO_size(1)*pars.reco.RECO_size(2)*nslices*dim4;
        if expected_data > length(fid)
          fid_prime=fid; fid_prime(expected_data) = 0;
        elseif expected_data < length(fid)
          fid_prime = fid(1:expected_data);
        else
          fid_prime = fid;
        end
        fid = reshape(fid_prime, pars.reco.RECO_size(1), pars.reco.RECO_size(2), nslices, dim4);
        
        dim1 = pars.reco.RECO_fov(1)*10; % [mm]
        dim2 = pars.reco.RECO_fov(2)*10; % [mm]
        
        if length(pars.acqp.ACQ_slice_offset) == nslices
          % RARE - multiple slices           
          dim3 = (max(pars.acqp.ACQ_slice_offset)-min(pars.acqp.ACQ_slice_offset)) *(nslices/(nslices-1)); % [cm]
        else
          % FLASH-3D all image is one slice
          dim3 = pars.acqp.ACQ_slice_thick; % [mm]
          
        dim1 = pars.reco.RECO_fov(1)*10; % [mm]
        dim2 = pars.reco.RECO_fov(2)*10; % [mm]
        end
        
        % Bring this ot Chad's stadard
        fid = permute(fid, [2,1,3,4]);
        fid = flip(fid, 1);
        dim1a = dim1; dim1 = dim2; dim2 = dim1a;

        return_structure.Size  = [dim1, dim2, dim3, dim4];
    end
  case 'raw'    
    acqp_file = fullfile(the_path, 'acqp');
    method_file = fullfile(the_path, 'method');
    pars.raw = epr_ReadBrukerAcqp(acqp_file);
    pars.method = epr_ReadBrukerAcqp(method_file);
    
    if isempty(strfind(pars.raw.BYTORDA, 'little'))
      byteorder = 'b';
    else
      byteorder = 'l';
    end
    
    fp=fopen(fid_file, 'r', byteorder);
    if (fp == -1)
      disp(['mrigMRI loader error: couldn''t open fid file "' FileName '".']);
      return
    end
    
    pars.raw.ACQ_size
    pars.raw.NSLICES
    pars.raw.ACQ_slice_offset
    
    opt_dtype = {'GO_32BIT_SGN_INT'};
    opt_val_dtype = {'int32'};
    dtype = get_option(pars.raw.GO_raw_data_format, opt_dtype, opt_val_dtype, '?');

    % Get data from file
    fid = fread(fp, dtype);
    fclose(fp);
    
    fid = reshape(fid, 2, pars.raw.ACQ_size(1)/2, pars.raw.ACQ_size(2), pars.raw.ACQ_size(3));
    fid = squeeze(fid(1,:,:,:) + 1i*fid(2,:,:,:));
end

if isfield(pars, 'acqp')
end

return_structure.Raw = fid;

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

