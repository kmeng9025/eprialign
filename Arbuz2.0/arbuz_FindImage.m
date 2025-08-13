% ARBUZ_FINDIMAGE  find image in the project
% output_list = arbuz_FindImage(hGUI, input_list, criterion, arg, output_fields)
% hGUI              - application handle
% input_list        - the scope of the search [string, array or cell array]
%   all/master/slave/+/v   - all/master images/slave images/ SELECTED images/ images SHOWN
%   [MasterIdx, SlaveIdx]  - indexes of an image [int]
%   cell array of element [structure]
%      [].ImageIdx - master image index
%      [].Image    - master image name
%      [].SlaveIdx - slave image index (-1 for master)
%      [].Slave    - slave image name
%   either ImageIdx or Image has to be provided
%   either SlaveIdx or Slave has to be provided 
% criterion         - search criterion
%      Highlighted/Selected/Visible/Name/ImageType or '' [string]
% arg               - value of the field or '' [string]
% output_fields     - cell array of field names to be included into output_list [cell array of strings]
%                     NAME/FULLNAME/FILENAME/IMAGETYPE/SELECTED/DATA/MASK/SLAVELIST/BBOX/
%                     COLOR/SELCTEDCOLOR/NOTSELECTEDCOLOR/MasterType
%                     A/Apre/Aprime/Anext/Acurrent/Ashow/Anative/Aall
% output_list       - cell array that has structure (see input_list:5)
%                     fields 'ImageType' and 'Selected' are returned by default
% EXAMPLE: output_list = arbuz_FindImage(hh, 'all', 'ImageType', '2D', {'A'})

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function output_list = arbuz_FindImage(hGUI, input_list, criterion, arg, output_fields)


%Sample Output fields = {'Name','DATA', 'Filename','Ashow','bbox','Anative'};

% load the project
prj = getappdata(hGUI, 'project');

output_list = {};
if strcmpi(criterion, 'HIGHLIGHTED')
  for ii=1:size(prj.highlighted,1)
    image.ImageIdx = prj.highlighted(ii,1);
    image.SlaveIdx = prj.highlighted(ii,2);
    output_list{end+1} = image;
  end
else
  input_list = arbuz_ImageList(hGUI, input_list);
  
  for ii = 1:length(input_list)
    % string definition of the field's name
    if ~isfield(input_list{ii}, 'ImageIdx') && isfield(input_list{ii}, 'Image')
      input_list{ii}.ImageIdx = arbuz_GetImageIdxByName(prj,input_list{ii}.Image);
    end
    ImageIdx = input_list{ii}.ImageIdx;
    if ~isfield(input_list{ii}, 'SlaveIdx') && isfield(input_list{ii}, 'Slave')
      if isempty(input_list{ii}.Slave), input_list{ii}.SlaveIdx = -1;
      else
        input_list{ii}.SlaveIdx = arbuz_GetProxyIdxByName(prj,ImageIdx,input_list{ii}.Proxy);
      end
    end
    SlaveIdx = safeget(input_list{ii}, 'SlaveIdx', -1); 

    if ImageIdx < 1 || ImageIdx > length(prj.images), continue; end
    switch upper(criterion)
      case 'IMAGETYPE'
        if SlaveIdx > 0
          arg2 = prj.images{ImageIdx}.slaves{SlaveIdx}.ImageType;
        else
          arg2 = prj.images{ImageIdx}.ImageType;
        end
        if strcmp(arg2, arg)
          slaves.ImageIdx = ImageIdx;
          slaves.SlaveIdx = SlaveIdx;
          output_list{end+1} = slaves;
        end
      case 'NAME'
        if SlaveIdx > 0
          arg2 = prj.images{ImageIdx}.slaves{SlaveIdx}.Name;
        else
          arg2 = prj.images{ImageIdx}.Name;
        end
        if strcmp(arg2, arg)
          slaves.ImageIdx = ImageIdx;
          slaves.SlaveIdx = SlaveIdx;
          output_list{end+1} = slaves;
        end
      case 'UPPERCASENAME'
        if SlaveIdx > 0
          arg2 = upper(prj.images{ImageIdx}.slaves{SlaveIdx}.Name);
        else
          arg2 = upper(prj.images{ImageIdx}.Name);
        end
        if strcmp(arg2, arg)
          slaves.ImageIdx = ImageIdx;
          slaves.SlaveIdx = SlaveIdx;
          output_list{end+1} = slaves;
        end
      case 'INNAME'
        if SlaveIdx >0
          arg2 = prj.images{ImageIdx}.slaves{SlaveIdx}.Name;
        else
          arg2 = prj.images{ImageIdx}.Name;
        end
        if ~isempty(strfind(upper(arg2), upper(arg)))
          slaves.ImageIdx = ImageIdx;
          slaves.SlaveIdx = SlaveIdx;
          output_list{end+1} = slaves;
        end
      case 'FULLNAME'
        if SlaveIdx > 0,
          arg2 = sprintf('%s / %s', prj.images{ImageIdx}.Name,...
            prj.images{ImageIdx}.slaves{SlaveIdx}.Name);
        else
          arg2 = prj.images{ImageIdx}.Name;
        end
        if strcmp(arg2, arg)
          slaves.ImageIdx = ImageIdx;
          slaves.SlaveIdx = SlaveIdx;
          output_list{end+1} = slaves;
        end
      case 'FILENAME'
        if SlaveIdx == -1 && strcmp(prj.images{ImageIdx}.FileName, arg)
          slaves.ImageIdx = ImageIdx;
          slaves.SlaveIdx = SlaveIdx;
          output_list{end+1} = slaves;
        end
      case 'SELECTED'
        if SlaveIdx > 0,
          arg2 = prj.images{ImageIdx}.slaves{SlaveIdx}.Selected;
        else
          arg2 = prj.images{ImageIdx}.Selected;
        end
        if arg2
          slaves.ImageIdx = ImageIdx;
          slaves.SlaveIdx = SlaveIdx;
          output_list{end+1} = slaves;
        end
      case 'VISIBLE'
        if SlaveIdx > 0,
          arg2 = prj.images{ImageIdx}.slaves{SlaveIdx}.Visible;
        else
          arg2 = prj.images{ImageIdx}.Visible;
        end
        if arg2
          slaves.ImageIdx = ImageIdx;
          slaves.SlaveIdx = SlaveIdx;
          output_list{end+1} = slaves;
        end
      case 'LINK'
        if SlaveIdx == -1 && isequal(safeget(prj.images{ImageIdx}, 'Link', ''),arg)
          slaves.ImageIdx = ImageIdx;
          slaves.SlaveIdx = -1;
          output_list{end+1} = slaves;
        end
      case 'HASSLAVENAME'
         if SlaveIdx == -1 
              ProxyList = {};
              Count = 0;
            for jj=1:length(prj.images{ImageIdx}.slaves);
              ProxyList{end+1}.SlaveName = prj.images{ImageIdx}.slaves{jj}.Name;
              ProxyList{end}.SlaveIdx = jj;
              if ~isempty(strfind(upper(ProxyList{end}.SlaveName),(upper(arg))));
                  Count = Count+1;
              end
            end
            if Count ~= 0
            slaves.ImageIdx = ImageIdx;
            slaves.SlaveIdx = SlaveIdx;
            output_list{end+1} = slaves;
            end    
          end
      case 'FINDSLAVESWITHINNAME'
          if SlaveIdx == -1 
              ProxyList = {};
              for jj=1:length(prj.images{ImageIdx}.slaves)
                  if ~isempty(strfind(upper(prj.images{ImageIdx}.slaves{jj}.Name),upper(arg)))
              ProxyList{end+1}.SlaveName = prj.images{ImageIdx}.slaves{jj}.Name;
              ProxyList{end}.SlaveIdx = jj;
              ProxyList{end}.ImageIdx = ImageIdx;
                  end
              end         
        output_list = ProxyList;
          end
      case 'FINDSLAVESIMAGETYPE' %use when input list contains pure slaves
          if SlaveIdx > 0 ;            
              if ~isempty(strfind(lower(prj.images{ImageIdx}.slaves{SlaveIdx}.ImageType), lower(arg)));
                slaves.SlaveName =  prj.images{ImageIdx}.slaves{SlaveIdx}.Name;
                slaves.ImageIdx = ImageIdx;
                slaves.SlaveIdx = SlaveIdx;
                output_list{end+1} = slaves;
              end
          end
    
      case 'ISSIZE'
            if SlaveIdx == -1,
            arg2 = size(prj.images{ImageIdx}.data);
            if sum(arg==arg2)==3
          slaves.ImageIdx = ImageIdx;
          slaves.SlaveIdx = SlaveIdx;
          output_list{end+1} = slaves;
                
            end
            end
      case ''
        slaves.ImageIdx = ImageIdx;
        slaves.SlaveIdx = SlaveIdx;
        output_list{end+1} = slaves;
    end
  end
end

% add fields of request
for ii=1:length(output_list)
  output_list{ii}.Image = prj.images{output_list{ii}.ImageIdx}.Name;
  if output_list{ii}.SlaveIdx > 0
    output_list{ii}.Slave = prj.images{output_list{ii}.ImageIdx}.slaves{output_list{ii}.SlaveIdx}.Name;
    im = prj.images{output_list{ii}.ImageIdx}.slaves{output_list{ii}.SlaveIdx};
  else
    output_list{ii}.Slave = '';
    im = prj.images{output_list{ii}.ImageIdx};
  end
  output_list{ii}.ImageType = im.ImageType;
  output_list{ii}.Selected = safeget(im, 'Selected', 0);
  output_list{ii}.Visible = safeget(im, 'Visible', 0);

  for kk=1:length(output_fields)
    switch upper(output_fields{kk})
      case {'NAME','UPPERCASENAME'}
        output_list{ii}.Name = im.Name;
      case 'FULLNAME'
        if output_list{ii}.SlaveIdx > 0
          output_list{ii}.FullName = sprintf('%s / %s', prj.images{output_list{ii}.ImageIdx}.Name,im.Name);
        else
          output_list{ii}.FullName = im.Name;
        end
      case 'FILENAME'
        output_list{ii}.FileName = safeget(im, 'FileName', '');
      case 'ISSTORE'
        output_list{ii}.isStore = safeget(im, 'isStore', 1);
      case 'DATA'
% Recovery of the lost coordinates
%         if strcmp(im.ImageType, 'XYZ') && isempty(im.data)
%           res = regexp(im.Name, 'A(?<a1>\d+)_(?<a2>\d+)', 'names');
%           im.data = [str2num(res.a1), str2num(res.a2), 1];
%           arbuz_SetImage(hGUI, output_list(ii), 'data', im.data);
%         end
        % load image if it is not loaded yet
        %BE
        if ~im.isLoaded && ~isempty(im.FileName)
          try
            fprintf('Loading %s\n', im.FileName);
            [im.data, out_pars] = arbuz_LoadImage(im.FileName, im.ImageType);
            arbuz_ShowMessage(hGUI, sprintf('%s is loaded.', im.FileName));
            arbuz_SetImage(hGUI, output_list(ii), 'data', im.data);
            %   arbuz_SetImage(hGUI, output_list(ii), 'Mask', im.Mask);
            arbuz_UpdateInterface(hGUI);
          catch
            fprintf('Loading %s is failed\n', im.FileName);
          end
        end
        output_list{ii}.data =  safeget(im, 'data', []);
      case 'MASK'
        output_list{ii}.Mask = safeget(safeget(im, 'data_info', []), 'Mask', []);
      case {'BOX','BBOX'}
        output_list{ii}.Box =  safeget(im, 'box', size(safeget(im, 'data', [])));
      case 'A'
        output_list{ii}.A =  safeget(prj.images{output_list{ii}.ImageIdx}, 'A', eye(4));
      case 'COLOR'
        if output_list{ii}.Selected
          output_list{ii}.Color =  safeget(im, 'SelectedColor', safeget(im, 'NotSelectedColor', []));
        else
          output_list{ii}.Color =  safeget(im, 'NotSelectedColor', []);
        end
      case 'SELECTEDCOLOR'
        output_list{ii}.SelectedColor =  safeget(im, 'SelectedColor', []);
      case 'NOTSELECTEDCOLOR'
        output_list{ii}.NotSelectedColor =  safeget(im, 'NotSelectedColor', []);
      case 'MASTERTYPE'
        if output_list{ii}.SlaveIdx > 0
          output_list{ii}.MasterType = prj.images{output_list{ii}.ImageIdx}.ImageType;
        else
          output_list{ii}.MasterType = output_list{ii}.ImageType;
        end        
      case 'ANATIVE'
        output_list{ii}.Anative =  safeget(prj.images{output_list{ii}.ImageIdx}, 'Anative', eye(4));
      case 'ISLOADED'
        output_list{ii}.isLoaded =  safeget(prj.images{output_list{ii}.ImageIdx}, 'isLoaded', 0);
      case 'APRE'
        if output_list{ii}.SlaveIdx > 0
          Aslave =  safeget(im, 'A', eye(4));
        else
          Aslave = eye(4);
        end
        output_list{ii}.Apre =  Aslave*safeget(prj.images{output_list{ii}.ImageIdx}, 'Apre', eye(4));
      case 'APRIME'
        output_list{ii}.Aprime =  safeget(prj.images{output_list{ii}.ImageIdx}, 'Aprime', eye(4));
      case 'ANEXT'
        output_list{ii}.Anext =  safeget(prj.images{output_list{ii}.ImageIdx}, 'Anext', eye(4));
      case 'ASLAVE'
        if output_list{ii}.SlaveIdx > 0
          output_list{ii}.Aslave =  safeget(im, 'A', eye(4));
        else
          output_list{ii}.Aslave = eye(4);
        end
      case 'ACURRENT'
        if output_list{ii}.SlaveIdx > 0
          Aslave =  safeget(im, 'A', eye(4));
        else
          Aslave = eye(4);
        end
        Apre   =  safeget(prj.images{output_list{ii}.ImageIdx}, 'Apre', eye(4));
        A      =  safeget(prj.images{output_list{ii}.ImageIdx}, 'A', eye(4));
        Aprime =  safeget(prj.images{output_list{ii}.ImageIdx}, 'Aprime', eye(4));
        if numel(Apre) ~= 16
            error("Apre is not a rotation matrix");
        end
        if numel(A) ~= 16
            error("A is not a rotation matrix");
        end
        if numel(Aprime) ~= 16
            error("Aprime is not a rotation matrix");
        end
        output_list{ii}.Acurrent  =  Aslave*Apre*A*Aprime;
      case 'ASHOW'
        if output_list{ii}.SlaveIdx > 0
          Aslave =  safeget(im, 'A', eye(4));
        else
          Aslave = eye(4);
        end
        Apre   =  safeget(prj.images{output_list{ii}.ImageIdx}, 'Apre', eye(4));
        A      =  safeget(prj.images{output_list{ii}.ImageIdx}, 'A', eye(4));
        
        Aprime =  safeget(prj.images{output_list{ii}.ImageIdx}, 'Aprime', eye(4));
        Anext  =  safeget(prj.images{output_list{ii}.ImageIdx}, 'Anext', eye(4));
        if size(A,1)  ~=  size(Aprime,1) ; A  = eye(4); end %I guess A can be set to a vector? -MM
        output_list{ii}.Ashow  =  Aslave*Apre*A*Aprime*Anext;
      case 'AALL'
        if output_list{ii}.SlaveIdx == -1
          results = {};
          Transformations = arbuz_get(hGUI, 'Transformations');
          for tt=1:length(Transformations)
            for jj=1:length(Transformations{tt}.Matrices)
              if strcmp(Transformations{tt}.Matrices{jj}.Image, output_list{ii}.Image)
                results{end+1}.Transformation = Transformations{tt}.Name;
                results{end}.Matrice = Transformations{tt}.Matrices{jj}.A;
              end
            end
          end
          output_list{ii}.Aall = results;
        end
      case 'SLAVELIST'
        if output_list{ii}.SlaveIdx == -1
          ProxyList = {};
          if output_list{ii}.SlaveIdx <= 1
            for jj=1:length(prj.images{output_list{ii}.ImageIdx}.slaves)
              ProxyList{end+1}.ImageIdx = output_list{ii}.ImageIdx;
              ProxyList{end}.SlaveName = prj.images{output_list{ii}.ImageIdx}.slaves{jj}.Name;
              ProxyList{end}.SlaveIdx = jj;
            end
          end
          output_list{ii}.SlaveList = ProxyList;
        else
          output_list{ii}.SlaveList = {};
        end
      case 'LINK'
        output_list{ii}.Link = safeget(prj.images{output_list{ii}.ImageIdx}, 'Link', '');
      case 'LinkTransformation'
        output_list{ii}.LinkTransformation = safeget(prj.images{output_list{ii}.ImageIdx}, 'LinkTransformation', -1);
      case 'DICOM'
        data_info = safeget(prj.images{output_list{ii}.ImageIdx}, 'data_info', []);
        output_list{ii}.DICOM = safeget(data_info, 'dicom_pars', []);
      case 'TAG1'
        output_list{ii}.Tag1 = safeget(im, 'Tag1', '');
      case 'TAG2'
        output_list{ii}.Tag1 = safeget(im, 'Tag2', '');
      case 'TAG3'
        output_list{ii}.Tag1 = safeget(im, 'Tag3', '');
      case 'GRID'
        Box =  safeget(im, 'box', size(safeget(im, 'data', [])));
        Anative =  safeget(prj.images{output_list{ii}.ImageIdx}, 'Anative', eye(4));
        for pp=1:3
          pmin  = [0 0 0 1]; pmax  = pmin;
          pmin(pp)=1; pmax(pp)=Box(pp);
          delta = pmax*Anative - pmin*Anative;
          delta = max(abs(delta));
          d{pp} = linspace(-delta/2, delta/2, Box(pp));
        end
        [output_list{ii}.gridX, output_list{ii}.gridY, output_list{ii}.gridZ]=meshgrid(d{1},d{2},d{3});
      otherwise 
        arbuz_ShowMessage(hGUI, ['arbuz_FindImage: unknown output field ''',output_fields{kk},'''.']);
    end
  end
end
