% --------------------------------------------------------------------
function [fout, out_pars, ftype, slave_images] = arbuz_LoadImage(fname, ftype)
slave_images = {};

fprintf('Loading image from ''%s''...\n', fname);
out_pars = [];
fout = [];
try
  switch ftype
    case '2D'
      fout = imread(fname);
      sz = size(fout);
      out_pars.Bbox    = [sz(1), sz(2), 0];
      out_pars.Anative = hmatrix_translate(-out_pars.Bbox/2);
      file = dir(fname);
      out_pars.DateTime = file.date;
    case '3DEPRI'
      [~, ~, ffext] = fileparts(fname);
      switch strtrim(ffext)
        case {'.IMG', '.img'}
        case {'.mat', '.MAT'}
          s = epr_LoadMATFile(fname, true, {'RAW3D'});
          fout = s.Raw;
          out_pars.Bbox = size(fout);
          out_pars.Anative = hmatrix_translate(-out_pars.Bbox/2)* ...
            hmatrix_scale(10*s.Size./out_pars.Bbox);
          file = dir(fname);
          out_pars.DateTime = file.date;
      end
    case 'pEPROI'
      answer = questdlg('Load RAW image?', 'Load options', 'Yes-First', 'Yes-Last', 'No', 'No');
      if strcmp(answer, 'No')
        s = epr_LoadMATFile(fname, true, {'pO2', 'Amp'});
      else
        s = epr_LoadMATFile(fname, true, {'pO2', 'Amp', 'Raw'});
      end
      fout = s.pO2;
      fout(~s.Mask) = -100;
      out_pars.Bbox = size(fout);
      out_pars.Anative = hmatrix_translate(-out_pars.Bbox/2)* ...
        hmatrix_scale(10*s.Size./out_pars.Bbox);
      out_pars.Mask = s.Mask;
      file = dir(fname);
      out_pars.DateTime = file.date;
      
      slave_images{end+1}.data = s.Amp;
      slave_images{end}.A = eye(4);
      slave_images{end}.Name = 'Amp';
      slave_images{end}.ImageType = 'pEPROI';

      slave_images{end+1}.data = s.Mask;
      slave_images{end}.A = eye(4);
      slave_images{end}.Name = 'Mask';
      slave_images{end}.ImageType = '3DMASK';
      
      switch answer
%         case 'Yes-4D'
%           slave_images{end+1}.data = s.Raw;
%           slave_images{end}.Name = 'Raw';
%           slave_images{end}.A = eye(4);
%           slave_images{end}.ImageType = '3DEPRI';
        case 'Yes-First'
          slave_images{end+1}.data = s.Raw(:,:,:,1);
          slave_images{end}.Name = 'Raw';
          slave_images{end}.A = eye(4);
          slave_images{end}.ImageType = '3DEPRI';
        case 'Yes-Last'
          slave_images{end+1}.data = s.Raw(:,:,:,end);
          slave_images{end}.Name = 'Raw';
          slave_images{end}.A = eye(4);
          slave_images{end}.ImageType = '3DEPRI';
      end
    case 'PO2_pEPRI'
      % cw p-file data
      s = epr_LoadMATFile(fname, true, {'pO2'});
      
      fout = s.pO2;
      fout(~s.Mask) = -100;
      out_pars.Bbox = size(fout);
      out_pars.Anative = hmatrix_translate(-out_pars.Bbox/2)* ...
        hmatrix_scale(10*s.Size./out_pars.Bbox);
      out_pars.Mask = s.Mask;
      file = dir(fname);
      out_pars.DateTime = file.date;
    case 'AMP_pEPRI'
      % cw p-file data
      s = epr_LoadMATFile(fname, true, {'CC'});
      
      fout = s.Amp;
      fout(~s.Mask) = 0;
      out_pars.Bbox = size(fout);
      out_pars.Anative = hmatrix_translate(-out_pars.Bbox/2)* ...
        hmatrix_scale(10*s.Size./out_pars.Bbox);
      file = dir(fname);
      out_pars.DateTime = file.date;
    case 'JIVATDMS'
      s = epr_LoadMATFile(fname, true, {'pO2'});
      
      if isfield(s, 'I3D')
        fout = s.I3D;
        out_pars.Bbox = size(fout);
        out_pars.Anative = hmatrix_translate(-out_pars.Bbox/2)* ...
          hmatrix_scale(10*s.Size./out_pars.Bbox);
        file = dir(fname);
        out_pars.DateTime = file.date;
      else
        fout = s.pO2;
        fout(~s.Mask) = 0;
        out_pars.Bbox = size(fout);
        out_pars.Anative = hmatrix_translate(-out_pars.Bbox/2)* ...
          hmatrix_scale(10*s.Size./out_pars.Bbox);
        file = dir(fname);
        out_pars.DateTime = file.date;
      end
    case 'MRI'
      [~,~,ext] = fileparts(fname);
      switch upper(ext)
        case '.MAT'
          [mriImg1,userbrowsed]=pickVariable(0,2,fname,'Pick Img & Info');
          idx_img = 1; idx_dsc = 2;
          if ~contains(char(mriImg1{1}), 'Img'), idx_img = 2; idx_dsc = 1; end
          if (userbrowsed == 0)
            si = load(fname,mriImg1{idx_img});
            sp = load(fname,mriImg1{idx_dsc});
          else  %else if not browsed
            si = eval(['load(''',userbrowsed,''', ''',char(mriImg1{idx_img}),''')']);
            sp = eval(['load(''',userbrowsed,''', ''',char(mriImg1{idx_dsc}),''')']);
          end %end if user changed file in pickVariable
          fout = getfield(si, char(mriImg1{idx_img}));
          out_pars.pars_out = getfield(sp, char(mriImg1{idx_dsc}));
          
          dx = out_pars.pars_out.sliceOffset(2)-out_pars.pars_out.sliceOffset(1);
          out_pars.Bbox = size(fout);
          out_pars.Anative = hmatrix_translate(-[(out_pars.Bbox(1:2)+1)/2, 3/2])* ...
            hmatrix_scale(out_pars.pars_out.scale.*[1,1,sign(dx)])*...
            hmatrix_translate([0, 0, out_pars.pars_out.sliceOffset(1)]);
        case {'.IMG', '.', ''}
          [s, pars] = epr_LoadBrukerMRI(fname);
          fprintf('ACQ_method: %s\n', pars.acqp.ACQ_method);
          fout = s.Raw;
          out_pars.Bbox = size(fout);
          scale = s.Size(1:3)./out_pars.Bbox(1:3);
          scale3 = scale(1:3);
          out_pars.Anative = hmatrix_translate(-out_pars.Bbox(1:3) / 2)* ...
            hmatrix_scale(scale3([2,1,3]));
          file = dir(fname);
          out_pars.DateTime = file.date;
      end
    case '3DSURFACE'
      [TumorEllipsoid, isOk] = TumorTouchEditDLG(fname);
      if isOk
        fout = TumorTouchToSurface(TumorEllipsoid);
        out_pars.Bbox = [1 1 1];
        out_pars.Anative = eye(4);
      end
    case 'RAW'
      res = inputdlg({'MatrixSize [k,l,m]', 'Dimensions [mm,mm,mm]', 'Stream (l/b)', 'Data (int16=>real/float=>real*4)'}, ...
        'Load parameters', 1, {'[100,100,100]','[1,1,1]','l','int16=>real'});
      matrix_size = str2num(res{1}); matrix_dim = str2num(res{2});
      fid = fopen(fname, 'r', res{3});
      fout = reshape(fread(fid, prod(matrix_size), res{4}), matrix_size);
      fclose(fid);
      out_pars.Anative = hmatrix_scale(matrix_dim);
      out_pars.Bbox = size(fout);
      out_pars.Anative = hmatrix_translate(-out_pars.Bbox/2)* ...
        hmatrix_scale(matrix_dim./out_pars.Bbox);
    case 'AMIRA3D'
      [out_pars.pars_out,fout]=read_amira_image_file(fname);
      out_pars.Anative = out_pars.pars_out.pixel_to_world;
      out_pars.Bbox = size(fout);
    case 'DICOM3D'
      [out_pars.pars_out,fout, out_pars.dicom_pars]=read_dicom_image_file(fname);
      
      %Translate the image to the treatment center.
      Meta = out_pars.dicom_pars;
      Treatment_center = Meta.ImagePositionPatient;
      
      out_pars.Anative = out_pars.pars_out.pixel_to_world;
      Manufacturer = safeget(out_pars.dicom_pars, 'Manufacturer', '');
      if contains(Manufacturer, 'Molecubes')
        RescaleIntercept = safeget(out_pars.dicom_pars,'RescaleIntercept',0);
        RescaleSlope = safeget(out_pars.dicom_pars,'RescaleSlope',1);
        fout = fout * RescaleSlope + RescaleIntercept;
      elseif any(Treatment_center)
        Origin = out_pars.pars_out.origin;
        Shift = (Treatment_center + Origin) * 10 * 2; %Doesn't read resolution. Assumes isotropic;
        fout = imtranslate(fout,Shift','OutputView','full');
        out_pars.Bbox = size(fout);
      end
      file = dir(fname);
      out_pars.DateTime = file.date;
    case 'SHAPE3D'
      disp('Shape loading is not supported.');
      fout = []; out_pars.Bbox = [0,0,0];
      out_pars.Anative = eye(4);
    case 'GENERIC'
      s1 = load(fname);
      %       Foolish code from mm 5/21/15
      %       Num_hashes = numel(strfind(fname, '\'))
      %       for ii = [1:Num_hashes-1]
      %           [Prefix,Suffix] = strtok(fname,'\')
      %           fname = Suffix
      %       end
      %
      %        dot_mat_idx = strfind(fname, '.mat')
      %
      
      
      fout = s1.data;
      out_pars.Bbox = s1.box;
      out_pars.Anative = s1.Anative;
      out_pars.generic_type = s1.ImageType;
    case 'FITRESULT'
      vars = evalin('base','whos');
      
      vlist = {};
      for ii=1:length(vars)
        if numel(find(vars(ii).size > 1)) >= 3
          vlist{end+1} = vars(ii).name;
        elseif strcmp(vars(ii).class, 'struct') && prod(vars(ii).size)
          strvars = evalin('base', ['fieldnames(',vars(ii).name,')']);
          for jj=1:length(strvars)
            elm = evalin('base', [vars(ii).name, '.', strvars{jj}]);
            if ndims(elm) >= 3
              vlist{end+1} = [vars(ii).name, '.', strvars{jj}];
            end
          end
        end
      end
      
      if ~isempty(vlist)
        [sel, ok] = listdlg('ListString', vlist);
        if ok
          fout = evalin('base', vlist{sel});
          out_pars.Bbox = size(fout);
          pos = strfind(vlist{sel}, '.');
          if pos
            % find Mask
            mask_matrix = vlist{sel};
            try
              mask_matrix = [mask_matrix(1:pos),'Mask'];
              out_pars.Mask = evalin('base', mask_matrix);
              disp('arbuz_LoadImage: Mask is found.');
            catch
            end
            % find Size
            try
              mask_matrix = [mask_matrix(1:pos),'rec_info'];
              rec_info = evalin('base', mask_matrix);
              Size = rec_info.rec.size;
              out_pars.Anative = hmatrix_translate(-out_pars.Bbox/2)* ...
                hmatrix_scale(10*Size./out_pars.Bbox);
              disp('arbuz_LoadImage: Size is found.');
            catch
            end
          end
        end
      else
        disp('No variables found.');
      end
    case 'IDL'
      A = restore_idl(fname);
      if isfield(A, 'KVMAPS2')
        fout = permute(A.KVMAPS2, [4,3,2,1]);
        % here we will get one slice
        sz = size(fout);
        selection = cell(sz(4), 1);
        for ii=1:sz(4)
          selection{ii} = sprintf('Slice %i', ii);
        end
        [s,v] = listdlg('PromptString','Select the slice:',...
          'SelectionMode','single',...
          'ListString',selection);
        if ~isempty(s)
          fout = fout(:,:,:,s);
        end
      elseif isfield(A, 'SE21')
        fout = permute(A.SE21, [3,2,1]);
      end
      scale = [1,1,1];
      offset = [0,0,0];
      [fp, fn] = fileparts(fname);
      fname = fullfile(fp, [fn,'.arbuz']);
      if exist(fname, 'file')
        ini = inimanage(fname);
        scale = eval(ini.T1.resolution);
        offset = eval(safeget(ini.T1, 'offset', '[0,0,0]'));
      end
      out_pars.Bbox = size(fout);
      out_pars.Anative = hmatrix_translate(-out_pars.Bbox(1:3) / 2)* ...
        hmatrix_scale(scale)*hmatrix_translate(offset);
    case 'WORKSPACE'
      vars = evalin('base','whos');
      disp('Select workspace variable:')
      disp('3D/4D Array')
      disp('   a dialog to enter scale will pop up')
      disp('Structure with fields:')
      disp('   data : 3D array')
      disp('   size : 1D array of image size i.e. [45,45,22]')
      disp('   type : string, i.e. MRI / 3DMASK / 3DEPRI')
      
      vlist = {};
      for ii=1:length(vars)
        if numel(find(vars(ii).size > 1)) == 2 || numel(find(vars(ii).size ==3))== 1 %include 3 channel images in the 2D catagory.
          vlist{end+1} = ['2D: ',vars(ii).name];
        elseif numel(find(vars(ii).size > 1)) == 3
          vlist{end+1} = ['3D: ',vars(ii).name];
        elseif numel(find(vars(ii).size > 1)) == 4
          vlist{end+1} = ['4D: ',vars(ii).name];
        elseif strcmp(vars(ii).class, 'struct') && prod(vars(ii).size)
          strvars = evalin('base', ['fieldnames(',vars(ii).name,')']);
          for jj=1:length(strvars)
            elm = evalin('base', [vars(ii).name, '.', strvars{jj}]);
            if ndims(elm) >= 3
              vlist{end+1} = [vars(ii).name, '.', strvars{jj}];
            end
          end
        end
      end
      
      if ~isempty(vlist)
        vlist = sort(vlist);
        [sel, ok] = listdlg('ListString', sort(vlist),'PromptString','See command window for help');
        if ok
          varname = vlist{sel};
          
          % this is structure
          if contains(varname, '.')
              pos = strfind(varname, '.');
              data_structure = evalin('base', varname(1:pos-1));
              fout = data_structure.data;
              out_pars.Bbox = size(fout);
              if isfield(data_structure, 'size')
                  out_pars.Size = data_structure.size;
              else
                  out_pars.Size = ones(ndims(fout));
              end
              if isfield(data_structure, 'type')
                  ftype = data_structure.type;
              end
          else % this is a matrix
              varname = varname(5:end);
              fout = evalin('base', varname);
              [scale, offset] = load_scale(fname);
              out_pars.Bbox = size(fout);
              while length(out_pars.Bbox) < 3, out_pars.Bbox = [out_pars.Bbox,1]; end
              out_pars.Size = scale;
          end
          
          out_pars.Anative = hmatrix_translate(-out_pars.Bbox + 1)* ...
            hmatrix_scale(out_pars.Size./out_pars.Bbox);
        end
      else
        disp('No variables found.');
      end
    case 'BIN'
      [DataFormat, Dims] = load_format(fname);
      
      fid = fopen(fname, 'rb');
      k = fread(fid, Dims(1)*Dims(2)*Dims(3), DataFormat);
      fout = reshape(k, Dims);
      fclose(fid);
      [scale, offset] = load_scale(fname);
      
      out_pars.Bbox = size(fout);
      out_pars.Anative = hmatrix_translate(-out_pars.Bbox(1:3) / 2)* ...
        hmatrix_scale(scale);
      
      save_scale(fname, scale, offset);
      save_format(fname, DataFormat, Dims);
    case 'MAT-GENERAL'
      s1 = load(fname);
      vars = fieldnames(s1);
      
      vlist = {};
      %       for ii=1:length(vars)
      %         if numel(find(vars(ii).size > 1)) == 2 || numel(find(vars(ii).size ==3))== 1 %include 3 channel images in the 2D catagory.
      %           vlist{end+1} = ['2D: ',vars(ii).name];
      %         elseif numel(find(vars(ii).size > 1)) == 3
      %           vlist{end+1} = ['3D: ',vars(ii).name];
      %         elseif numel(find(vars(ii).size > 1)) == 4
      %           vlist{end+1} = ['4D: ',vars(ii).name];
      %         elseif strcmp(vars(ii).class, 'struct') && prod(vars(ii).size)
      %           strvars = evalin('base', ['fieldnames(',vars(ii).name,')']);
      %           for jj=1:length(strvars)
      %             elm = evalin('base', [vars(ii).name, '.', strvars{jj}]);
      %             if ndims(elm) >= 3
      %               vlist{end+1} = [vars(ii).name, '.', strvars{jj}];
      %             end
      %           end
      %         end
      %       end
      vlist = vars;
      
      if ~isempty(vlist)
        vlist = sort(vlist);
        [sel, ok] = listdlg('ListString', sort(vlist));
        if ok
          varname = vlist{sel};
          
          fout = s1.(varname);
          [scale, offset] = load_scale(fname);
          
          out_pars.Bbox = size(fout);
          out_pars.Anative = hmatrix_translate(-out_pars.Bbox(1:3) / 2)* ...
            hmatrix_scale(scale);
          
          save_scale(fname, scale, offset);
          
        end
      else
        disp('No variables found.');
      end
    case 'CEST_ML'
      [fout, out_pars, slave_images] = arbuz_custom_loader_cest(fname);
    otherwise
      disp('Unknown image format.');
      fout = []; out_pars.Bbox = [0,0,0];
      out_pars.Anative = eye(4);
  end
catch err
  error(err.message);
end
% out_pars.Anative

function [scale, offset] = load_scale(fname)

scale = [1,1,1];
offset = [0,0,0];
[fp, fn] = fileparts(fname);
fname = fullfile(fp, [fn,'.arbuz']);
if exist(fname, 'file') == 2
  ini = inimanage(fname);
  T1 = safeget(ini, 'T1', []);
  scale = eval(safeget(T1,'resolution', '[1,1,1]'));
  offset = eval(safeget(ini.T1, 'offset', '[0,0,0]'));
else
  answer=inputdlg({'ScaleX', 'ScaleY', 'ScaleZ','OffsetX', 'OffsetY', 'OffsetZ'},...
    'Set image parameters',1,...
    {num2str(scale(1)), num2str(scale(2)),num2str(scale(3)),...
    num2str(offset(1)), num2str(offset(2)),num2str(offset(3))});
  
  if ~isempty(answer)
    scale(1) = eval(answer{1});
    scale(2) = eval(answer{2});
    scale(3) = eval(answer{3});
    offset(1) = eval(answer{4});
    offset(2) = eval(answer{5});
    offset(3) = eval(answer{6});
  end
end

function save_scale(fname,scale, offset)

[fp, fn] = fileparts(fname);
fname = fullfile(fp, [fn,'.arbuz']);
if exist(fname, 'file')
  ini = inimanage(fname);
end
ini.T1.resolution = ['[',sprintf('%f ', scale), ']'];
ini.T1.offset = ['[',sprintf('%f ', offset), ']'];
inimanage(fname, ini);

function [DataFormat, Dims] = load_format(fname)

DataFormat = 'float32';
Dims = [1,1,1];

[fp, fn] = fileparts(fname);
fname = fullfile(fp, [fn,'.arbuz']);
if exist(fname, 'file')
  ini = inimanage(fname);
  data = safeget(ini, 'data', []);
  DataFormat = safeget(data, 'format', 'float32');
  Dims = eval(safeget(data, 'dims', '[1,1,1]'));
else
  answer=inputdlg({'DataFormat', 'Dim-1', 'Dim-2','Dim-3'},...
    'Set image parameters',1,...
    {DataFormat, ...
    num2str(Dims(1)), num2str(Dims(2)),num2str(Dims(3))});
  
  if ~isempty(answer)
    DataFormat = answer{1};
    Dims(1) = eval(answer{2});
    Dims(2) = eval(answer{3});
    Dims(3) = eval(answer{4});
  end
end

function save_format(fname, DataFormat, Dims)

[fp, fn] = fileparts(fname);
fname = fullfile(fp, [fn,'.arbuz']);
if exist(fname, 'file')
  ini = inimanage(fname);
end
ini.data.format = DataFormat;
ini.data.dims = ['[',sprintf('%i ',Dims),']'];
inimanage(fname, ini);