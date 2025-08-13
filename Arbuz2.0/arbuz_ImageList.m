% ARBUZ_IMAGELIST  find image in the project
% image_list = arbuz_ImageList(alias)
% hGUI              - application handle
% alias             - image request [string, int array or cell of strings/int]
%   all/master/slave/+/v   - all/master images/slave images/ SELECTED images/ images SHOWN [string]
%   [MasterIdx] or [MasterIdx, SlaveIdx]  - indexes of an image [int]
%   {Master} or {Master, Slave}  - name of an image [string]
% image_list        - the result [cell array]
%   element of array [structure]
%      [].ImageIdx - master image index
%      [].Image    - master image name
%      [].SlaveIdx - slave image index (-1 for master)
%      [].Slave    - slave image name

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function image_list = arbuz_ImageList(hGUI, alias_list)

image_list = {};

element = [];

if isstruct(alias_list)
  element = alias_list;
elseif iscell(alias_list)
  if ~isempty(alias_list)
    % image described by {'image'} or {'image', 'slave}
    if ischar(alias_list{1})
      element.Image = alias_list{1};
      if length(alias_list) > 1
        if ischar(alias_list{2})
          element.Slave = alias_list{2};
        elseif isnumeric(alias_list{2})
          element.SlaveIdx = alias_list{2};
        else
          element.SlaveIdx = -1;
        end
      else
        element.SlaveIdx = -1;
      end
    elseif isnumeric(alias_list{1})
      element.ImageIdx = alias_list{1};
      if length(alias_list) > 1
        if ischar(alias_list{2})
          element.Slave = alias_list{2};
        elseif isnumeric(alias_list{2})
          element.SlaveIdx = alias_list{2};
        else
          element.SlaveIdx = -1;
        end
      else
        element.SlaveIdx = -1;
      end
    else
      % copy alias to the output
      image_list = alias_list;
    end
  end
  % image described by [1] or [1,1]
elseif isnumeric(alias_list)
  element.ImageIdx = alias_list(1);
  if length(alias_list) > 1
    element.SlaveIdx = alias_list(2);
  else
    element.SlaveIdx = -1;
  end
% global definitions
elseif ischar(alias_list)
	prj = getappdata(hGUI, 'project');
    switch alias_list
      case 'all'
        for ii = 1:length(prj.images)
          image_list{end+1}.ImageIdx = ii;
          image_list{end}.SlaveIdx = -1;
          for jj=1:length(prj.images{ii}.slaves)
            image_list{end+1}.ImageIdx = ii;
            image_list{end}.SlaveIdx = jj;
          end
        end
      case 'master'
        for ii = 1:length(prj.images)
            image_list{end+1}.ImageIdx = ii;
            image_list{end}.SlaveIdx = -1;
        end
      case 'slave'
        for ii = 1:length(prj.images)
          for jj=1:length(prj.images{ii}.slaves)
            image_list{end+1}.ImageIdx = ii;
            image_list{end}.SlaveIdx = jj;
          end
        end
      case '+'
        for ii = 1:length(prj.images)
          if prj.images{ii}.Selected
            image_list{end+1}.ImageIdx = ii;
            image_list{end}.SlaveIdx = -1;
            for jj=1:length(prj.images{ii}.slaves)
              image_list{end+1}.ImageIdx = ii;
              image_list{end}.SlaveIdx = jj;
            end
          end
        end
      case 'v'
        for ii = 1:length(prj.images)
          if prj.images{ii}.Visible
            image_list{end+1}.ImageIdx = ii;
            image_list{end}.SlaveIdx = -1;
          end
          for jj=1:length(prj.images{ii}.slaves)
            if safeget(prj.images{ii}.slaves{jj}, 'Visible', 0)
              image_list{end+1}.ImageIdx = ii;
              image_list{end}.SlaveIdx = jj;
            end
          end
        end
    end
    return;
end

if ~isempty(element)
  image_list = {element};
end

prj = getappdata(hGUI, 'project');
for ii=1:length(image_list)
  % Master image  
  if ~isfield(image_list{ii}, 'ImageIdx') && isfield(image_list{ii}, 'Image')
    image_list{ii}.ImageIdx = -1;
    for jj=1:length(prj.images)
      if strcmp(prj.images{jj}.Name, image_list{ii}.Image)
        image_list{ii}.ImageIdx = jj;
        break;
      end
    end
    if image_list{ii}.ImageIdx == -1
      arbuz_ShowMessage(hGUI, ['arbuz_ImageList: Image ''',image_list{ii}.Image,''' was not found.']);
      return;
    end
  end
  if ~isfield(image_list{ii}, 'Image')
    image_list{ii}.Image = prj.images{image_list{ii}.ImageIdx}.Name;
  end
  % Slave image
  if ~isfield(image_list{ii}, 'SlaveIdx') && ~isfield(image_list{ii}, 'Slave')
    image_list{ii}.SlaveIdx = -1;
    image_list{ii}.Slave = '';
  elseif isfield(image_list{ii}, 'SlaveIdx')
    if image_list{ii}.SlaveIdx == -1
      image_list{ii}.Slave = '';
    elseif length(prj.images{image_list{ii}.ImageIdx}.slaves) >= image_list{ii}.SlaveIdx
      image_list{ii}.Slave = prj.images{image_list{ii}.ImageIdx}.slaves{image_list{ii}.SlaveIdx}.Name;
    else
      image_list{ii}.Slave = '';
    end
  elseif isfield(image_list{ii}, 'Slave')
    image_list{ii}.SlaveIdx = -1;
    if ~isempty(image_list{ii}.Slave)
      slvs = prj.images{image_list{ii}.ImageIdx}.slaves;
      for jj=1:length(slvs)
        if strcmp(slvs{jj}.Name, image_list{ii}.Slave)
          image_list{ii}.SlaveIdx = jj;
          break;
        end
      end
      if image_list{ii}.SlaveIdx == -1
        arbuz_ShowMessage(hGUI, ['arbuz_ImageList: Dependent image ''',image_list{ii}.Slave,''' was not found.']);
        return;
      end
    end
  end
end
