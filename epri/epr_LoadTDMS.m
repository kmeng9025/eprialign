function tdms = epr_LoadTDMS(f_name, tdms)

%   This is the partial implementation of TDMS format for LFEPR
%   TDMS format is based on information provided by National Instruments
%   at:    http://zone.ni.com/devzone/cda/tut/p/id/5696

if ~exist('tdms', 'var'), tdms = []; end
if ischar(f_name)
  fid = fopen(f_name);
  if fid < 0, error(['epr_LoadTDMS: The file ',f_name,' can not be open.']); end
  k = fread(fid,'*uint8');
  fclose(fid);
  tdms = [];
elseif iscell(f_name)
  %
else
  k=reshape(uint8(f_name),1,length(f_name));
end

end_of_data = length(k);

i32se = @(x, offset) typecast(x(offset:offset+3), 'int32');

new_offset = 1;
ii_seg = 1;
while new_offset < end_of_data-4
%   fprintf('Segment %i\n', ii_seg);
  tdmsTAG = k(new_offset:new_offset+3); new_offset = new_offset + 4;
  if ~strcmp(char(tdmsTAG(:))', 'TDSm'), error(sprintf('epr_LoadTDMS: file tag is not recognized at position %i.',new_offset-4)); end
  tdms.TOC = uint32(i32se(k, new_offset)); new_offset = new_offset + 4;
  kTocMetaData=bitget(tdms.TOC,2);
  kTocNewObject=bitget(tdms.TOC,3);
  kTocRawData=bitget(tdms.TOC,4);
  opt.kTocBigEndian=bitget(tdms.TOC,7);
  raw_idx = {};
  
  if opt.kTocBigEndian %big-endian
    opt.i64 = @(x, offset) typecast(x(offset+7:-1:offset), 'int64');
    opt.i32 = @(x, offset) typecast(x(offset+3:-1:offset), 'int32');
    opt.i16 = @(x, offset) typecast(x(offset+1:-1:offset), 'int16');
    opt.af32 = @(x, offset, length) flipud(typecast(flipud(x(offset+(0:4*length-1))), 'single'));
    opt.af64 = @(x, offset, length) flipud(typecast(flipud(x(offset+(0:8*length-1))), 'double'));
    opt.ai8 = @(x, offset, length) flipud(typecast(flipud(x(offset+(0:length-1))), 'int8'));
    opt.ai16 = @(x, offset, length) flipud(typecast(flipud(x(offset+(0:4*length-1))), 'int16'));
    opt.ai32 = @(x, offset, length) flipud(typecast(flipud(x(offset+(0:4*length-1))), 'int32'));
    opt.ai64 = @(x, offset, length) flipud(typecast(flipud(x(offset+(0:8*length-1))), 'int64'));
  else % small-endian
    opt.i64 = @(x, offset) typecast(x(offset:offset+7), 'int64');
    opt.i32 = @(x, offset) typecast(x(offset:offset+3), 'int32');
    opt.i16 = @(x, offset) typecast(x(offset:offset+1), 'int16');
    opt.af32 = @(x, offset, length) typecast(x(offset+(0:4*length-1)), 'single');
    opt.af64 = @(x, offset, length) typecast(x(offset+(0:8*length-1)), 'double');
    opt.ai8 = @(x, offset, length) typecast(x(offset+(0:length-1)), 'int8');
    opt.ai16 = @(x, offset, length) typecast(x(offset+(0:4*length-1)), 'int16');
    opt.ai32 = @(x, offset, length) typecast(x(offset+(0:4*length-1)), 'int32');
    opt.ai64 = @(x, offset, length) typecast(x(offset+(0:8*length-1)), 'int64');
  end
  tdms.version = opt.i32(k, new_offset); new_offset = new_offset + 4;
  
  SEGMENT_LENGTH = opt.i64(k, new_offset); new_offset = new_offset + 8;
  NEXT_SEGMENT_OFFSET = new_offset + double(SEGMENT_LENGTH) + 8;
  RAWDATA_LENGTH = opt.i64(k, new_offset); new_offset = new_offset + 8;
  RAW_DATA_POS = new_offset + double(RAWDATA_LENGTH);
  
  if kTocMetaData
    NEWOBJ_NUMBER = opt.i32(k, new_offset); new_offset = new_offset + 4;
    
    for nobj = 1:NEWOBJ_NUMBER
      [new_obj_name, new_offset] = get_string(k, new_offset, opt);
      slash = strfind(new_obj_name(:)', '/');
      level = length(slash); % 1 or 2
      if isequal(new_obj_name, '/')
        Tree_position_1 = 'root';
        Tree_position_2 = '';
      elseif new_obj_name(1) == '/'
        if level == 1
          Tree_position_1 = new_obj_name(2:end);
          Tree_position_2 = '';
        else
          Tree_position_1 = new_obj_name(2:slash(2)-1);
          Tree_position_2 = new_obj_name(slash(2)+1:end);
        end
        Tree_position_1 = validMLfield(Tree_position_1);
        Tree_position_2 = validMLfield(Tree_position_2);
      end
      RAW_DATA_DESCRIPTOR_LENGTH = opt.i32(k, new_offset); new_offset = new_offset + 4;
      
      % Raw data header (-1 for no data header)
      if RAW_DATA_DESCRIPTOR_LENGTH == 20
        raw_idx{end+1}.RAW_DATA_TYPE = opt.i32(k, new_offset); new_offset = new_offset + 4;
        raw_idx{end}.RAW_DATA_ARRAY_DIM = opt.i32(k, new_offset); new_offset = new_offset + 4; % always 1
        raw_idx{end}.RAW_DATA_LENGTH = int32(opt.i64(k, new_offset)); new_offset = new_offset + 8;
        raw_idx{end}.RAW_DATA_TREE1 = Tree_position_1;
        raw_idx{end}.RAW_DATA_TREE2 = Tree_position_2;
      end
      
      % get the object
      an_obj = get_the_object(tdms, Tree_position_1, Tree_position_2);
      % read the object properties
      NUMBER_OF_PROPERTIES = opt.i32(k, new_offset); new_offset = new_offset + 4;
      for jj=1:NUMBER_OF_PROPERTIES
        [an_obj, new_offset] = get_property(k, new_offset, an_obj, opt);
      end
      % set the object
      if level == 1
        tdms.(Tree_position_1) = an_obj;
      else
        tdms.(Tree_position_1).(Tree_position_2) = an_obj;
      end
    end
    ii_seg = ii_seg + 1;
  end
  
  % Raw data reader
  if kTocRawData
    for raw_set=1:length(raw_idx)
      RAW_DATA_LENGTH = raw_idx{raw_set}.RAW_DATA_LENGTH;
      RAW_DATA_TREE1 = raw_idx{raw_set}.RAW_DATA_TREE1;
      RAW_DATA_TREE2 = raw_idx{raw_set}.RAW_DATA_TREE2;
      % only very limited subset of datatypes is implemented
      switch raw_idx{raw_set}.RAW_DATA_TYPE
        case 2 %int16
          data = opt.ai16(k,new_offset,RAW_DATA_LENGTH);
          new_offset = new_offset + 2*RAW_DATA_LENGTH;
        case 3 %int32
          data = opt.ai32(k,new_offset,RAW_DATA_LENGTH);
          new_offset = new_offset + 4*RAW_DATA_LENGTH;
        case 4 %int64
          data = opt.ai64(k,new_offset,RAW_DATA_LENGTH);
          new_offset = new_offset + 8*RAW_DATA_LENGTH;
        case 9 % float32
          data = opt.af32(k,new_offset,RAW_DATA_LENGTH);
          new_offset = new_offset + 4*RAW_DATA_LENGTH;
        case 10 % double64
          data = opt.af64(k,new_offset,RAW_DATA_LENGTH);
          new_offset = new_offset + 8*RAW_DATA_LENGTH;
        case 25 % float32
          data = opt.af32(k,new_offset,RAW_DATA_LENGTH);
          new_offset = new_offset + 4*RAW_DATA_LENGTH;
        case 26
          data = opt.af64(k,new_offset,RAW_DATA_LENGTH);
          new_offset = new_offset + 8*RAW_DATA_LENGTH;
%           unit = k(new_offset + int32(0:10))
        case 33 %boolean  
          data = boolean(opt.ai8(k,new_offset,RAW_DATA_LENGTH));
          new_offset = new_offset + RAW_DATA_LENGTH;
        case 68 % time
          binary_time = reshape(k(new_offset+(0:16*RAW_DATA_LENGTH-1)), 16, RAW_DATA_LENGTH);
          tmsec = zeros(RAW_DATA_LENGTH, 1);
          tsec = zeros(RAW_DATA_LENGTH, 1);
          for ii=1:RAW_DATA_LENGTH
            tmsec(ii) = opt.i64(binary_time(:, ii), 1);
            tsec(ii) = opt.i64(binary_time(:, ii), 9);
          end
          data = (double(tsec) + double(tmsec)/2^64)/86400+695422-5/24;
          new_offset = new_offset + 16*RAW_DATA_LENGTH;
        otherwise
          error(['Unknown data type ',num2str(raw_idx{raw_set}.RAW_DATA_TYPE), '.']);
      end
      if level == 1
        if ~isfield(tdms.(RAW_DATA_TREE1), 'data_index'), tdms.(RAW_DATA_TREE1).data_index = 0; end
        current_idx = tdms.(RAW_DATA_TREE1).data_index;
        tdms.(RAW_DATA_TREE1).data_index = current_idx + 1;
        tdms.(RAW_DATA_TREE1).data{current_idx+1} = data;
      else
        if ~isfield(tdms.(RAW_DATA_TREE1).(RAW_DATA_TREE2), 'data_index'), tdms.(RAW_DATA_TREE1).(RAW_DATA_TREE2).data_index = 0; end
        current_idx = tdms.(RAW_DATA_TREE1).(RAW_DATA_TREE2).data_index;
        tdms.(RAW_DATA_TREE1).(RAW_DATA_TREE2).data_index = current_idx + 1;
        if ~isempty(data)
         tdms.(RAW_DATA_TREE1).(RAW_DATA_TREE2).data{current_idx+1} = data;
        end
      end
    end
  end
  new_offset = NEXT_SEGMENT_OFFSET;
end

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

% --------------------------------------------------------------------
function [val, new_offset] = get_string(strm, offset, opt)
LENGTH = opt.i32(strm, offset); new_offset = offset + 4;
if LENGTH > 4000000
  error('Too long string (suspicious).');
end
val = char(strm(new_offset:new_offset + LENGTH-1)); new_offset = new_offset + LENGTH;
val = val(:)';

% --------------------------------------------------------------------
function [a, new_offset, is_finished] = get_property(strm, new_offset, a, opt)
is_finished = 0; old_offset = new_offset;
[field_name, new_offset]  = get_string(strm, new_offset, opt);
data_type = opt.i32(strm, new_offset); new_offset = new_offset + 4;
% only very limited subset of datatypes is implemented
if(data_type >= 2 && data_type <= 11)
  [matType, matLength]=LV2MatlabDataType(data_type);
  val_array = strm(new_offset:new_offset+matLength-1);
  if opt.kTocBigEndian %big-endian
    field_value = typecast(flipud(val_array), matType);
  else % small-endian
    field_value = typecast(val_array, matType);
  end
  new_offset = new_offset + matLength;
else
  switch data_type
    case -1,
      is_finished = 1; new_offset = old_offset; return;
    case 32, [field_value, new_offset]  = get_string(strm, new_offset, opt);
    case 33,
      field_value = strm(new_offset) ~= 0; new_offset = new_offset + 1;
    case 68, % time stamp 128
      length = 16;
      binary_time = strm(new_offset:new_offset + length - 1)'; new_offset = new_offset + length;
      tmsec = opt.i64(binary_time, 1);
      tsec = opt.i64(binary_time, 9);
      field_value = (double(tsec) + double(tmsec)/2^64)/86400+695422-5/24;
    otherwise,
      disp(data_type);
      error('Unknown data type');
  end
end
a.(validMLfield(field_name)) = field_value;

% --------------------------------------------------------------------
function [a, new_offset, is_finished] = get_property_echo(strm, new_offset, a, opt)
is_finished = 0; old_offset = new_offset;
[field_name, new_offset]  = get_string(strm, new_offset, opt);
data_type = opt.i32(strm, new_offset); new_offset = new_offset + 4;
% only very limited subset of datatypes is implemented
if(data_type >= 2 && data_type <= 11)
  [matType, matLength]=LV2MatlabDataType(data_type);
  val_array = strm(new_offset:new_offset+matLength-1);
  if opt.kTocBigEndian %big-endian
    field_value = typecast(flipud(val_array), matType);
  else % small-endian
    field_value = typecast(val_array, matType);
  end
  new_offset = new_offset + matLength;
else
  switch data_type
    case -1
      is_finished = 1; new_offset = old_offset; return;
    case 32 
      [field_value, new_offset]  = get_string(strm, new_offset, opt);
      fprintf('object=%s offset=%d type=%d strval=%s after(50)=%s\n', field_name, new_offset, data_type, field_value, print_char(strm(new_offset:new_offset+50)'));
    case 33
      field_value = strm(new_offset) ~= 0; new_offset = new_offset + 1;
    case 68 % time stamp 128
      length = 16;
      binary_time = strm(new_offset:new_offset + length - 1)'; new_offset = new_offset + length;
      tmsec = opt.i64(binary_time, 1);
      tsec = opt.i64(binary_time, 9);
      field_value = (double(tsec) + double(tmsec)/2^64)/86400+695422-5/24;
    otherwise
      disp(data_type);
      error('Unknown data type');
  end
end
a.(validMLfield(field_name)) = field_value;

%
function out = print_char(in)
out = '';
for ii=1:length(in)
  if isstrprop(in(ii), 'alphanum') out = [out, in(ii),'|'];
  elseif isstrprop(in(ii), 'punct') out = [out, in(ii),'|'];
  elseif in(ii) == '\' out = [out, in(ii),'|'];
  elseif in(ii) == '_' out = [out, in(ii),'|'];
  elseif in(ii) == ' ' out = [out, in(ii),'|'];
  else
    out = [out, dec2hex(in(ii)),'|'];
  end
end

% --------------------------------------------------------------------
function [matType, matLength]=LV2MatlabDataType(LVType)
%Cross Refernce Labview TDMS Data type to MATLAB
switch LVType
  case 0   %tdsTypeVoid
    matType=''; matLength = 0;
  case 1   %tdsTypeI8
    matType='int8'; matLength = 1;
  case 2   %tdsTypeI16
    matType='int16'; matLength = 2;
  case 3   %tdsTypeI32
    matType='int32'; matLength = 4;
  case 4   %tdsTypeI64
    matType='int64'; matLength = 8;
  case 5   %tdsTypeU8
    matType='uint8'; matLength = 1;
  case 6   %tdsTypeU16
    matType='uint16'; matLength = 2;
  case 7   %tdsTypeU32
    matType='uint32'; matLength = 4;
  case 8   %tdsTypeU64
    matType='uint64'; matLength = 8;
  case 9  %tdsTypeSingleFloat
    matType='single'; matLength = 4;
  case 10  %tdsTypeDoubleFloat
    matType='double'; matLength = 8;
  case 11  %tdsTypeExtendedFloat
    matType=''; matLength = 0;
  case 32  %tdsTypeString
    matType='char'; matLength = 0;
  case 33  %tdsTypeBoolean
    matType='logical'; matLength = 1;
  case 68  %tdsTypeTimeStamp
    matType='bit224'; matLength = 16;
  otherwise
    matType='';  matLength = 0;
end

% --------------------------------------------------------------------
function str = validMLfield(str)
symbols_to_kill = '/.[]?@#$():''-';
for ii=1:length(symbols_to_kill), str(str==symbols_to_kill(ii)) = []; end % remove symbols
str(str==' ') = '_'; % convert spaces into underscores
if isempty(str), str = 'Fempty'; 
elseif ~isempty(strfind('0123456789', str(1))), str = ['F',str];
end
str = str(:)';

% --------------------------------------------------------------------
function an_obj = get_the_object(tdms, fld_1, fld_2)
if ~isfield(tdms, fld_1), tdms.(fld_1) = []; end
if isempty(fld_2)
  an_obj = tdms.(fld_1);
else
  if isfield(tdms.(fld_1), fld_2),
    an_obj = tdms.(fld_1).(fld_2);
  else
    an_obj = [];
  end
end