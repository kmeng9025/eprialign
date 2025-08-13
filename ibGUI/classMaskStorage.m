classdef classMaskStorage
  properties
    hGUI
    options
    Type
  end
  % --------------------------------------------------------------------
  methods
    function obj=classMaskStorage(hGUI, Type, options)
      obj.hGUI = hGUI;
      obj.options.isPreserveDataMask = safeget(options, 'isPreserveDataMask', false);
      obj.options.isRestrict = safeget(options, 'isRestrict', false);
      obj.Type = Type;
    end
    % --------------------------------------------------------------------
    function res = n(obj)
      Masks = getappdata(obj.hGUI, obj.Type);
      res = length(Masks);
    end
    % --------------------------------------------------------------------
    function res = is(obj, idx)
      Masks = getappdata(obj.hGUI, obj.Type);
      res = idx <= length(Masks);
    end
    % --------------------------------------------------------------------
    function res = are(obj)
      Masks = getappdata(obj.hGUI, obj.Type);
      res = ~isempty(Masks);
    end
    % --------------------------------------------------------------------
    function Masks = all(obj)
      Masks = getappdata(obj.hGUI, obj.Type);
    end
    % --------------------------------------------------------------------
    function Mask = Get(obj, idx)
      Masks = getappdata(obj.hGUI, obj.Type);
      if idx < 1 || idx > length(Masks)
        Mask = [];
      else
        Mask = Masks{idx}.Mask;
      end
    end
    % --------------------------------------------------------------------
    % idx - if idx == -1 new entry is created, and idx is returned
    function idx = Set(obj, idx, Mask)
      if idx == 1 && obj.options.isPreserveDataMask
        warning('classMaskStorage::Set: Data mask was not modified. Uncheck data mask preservation.');
        return;
      end
      % Store new mask, remember previous mask (<5) for undo
      Masks = getappdata(obj.hGUI, obj.Type);
      if obj.options.isRestrict
        idxRestrict = obj.GetIdx('Restrict',[]);
        if ~isempty(idxRestrict)
          RestrictMask = obj.Get(idxRestrict);
          Mask = Mask & RestrictMask;
        end
      end
      if idx >= 1 && idx <= length(Masks)
        MaskUndo = getappdata(obj.hGUI, 'MaskUndo');
        for ii=min(5, length(MaskUndo)):-1:1
          MaskUndo{ii+1} = MaskUndo{ii};
        end
        MaskUndo{1} = Masks{idx};
        Masks{idx}.Mask = Mask;
        setappdata(obj.hGUI, 'MaskUndo', MaskUndo);
        setappdata(obj.hGUI, obj.Type, Masks);
      elseif idx == -1
        Masks{end+1}.Mask = Mask;
        idx = length(Masks);
        setappdata(obj.hGUI, obj.Type, Masks);
      else
        idx = -1;
      end
    end
    % --------------------------------------------------------------------
    function Delete(obj, idx)
      if ~obj.is(idx), return; end
      Masks = getappdata(obj.hGUI, obj.Type);
      Masks = Masks([1:idx-1,idx+1:end]);
      setappdata(obj.hGUI, obj.Type, Masks);
    end
    % --------------------------------------------------------------------
    % dims - if dims not empty new image will be created if 'Name' not found
    function idx = GetIdx(obj, Name, dims)
      Masks = getappdata(obj.hGUI, obj.Type);
      for ii=1:length(Masks)
        if isequal(Masks{ii}.Name, Name), idx = ii; return; end
      end
      if ~isempty(dims)
        Masks{end+1}.Name = Name;
        Masks{end}.Mask = false(dims);
        idx = length(Masks);
        setappdata(obj.hGUI, obj.Type, Masks);
      else
        idx = [];
      end
    end
    % --------------------------------------------------------------------
    function MaskList = MaskNames(obj)
      Masks = getappdata(obj.hGUI, obj.Type);
      MaskList =cell(length(Masks), 1);
      for ii=1:length(MaskList)
        MaskList{ii} = Masks{ii}.Name;
      end
    end
    % --------------------------------------------------------------------
    function Mask = GetEntry(obj, idx)
      Masks = getappdata(obj.hGUI, obj.Type);
      if idx < 1 || idx > length(Masks)
        Mask = [];
      else
        Mask = Masks{idx};
      end
    end
    % --------------------------------------------------------------------
    function Mask = SafeGet(obj, idx, image_dims)
      Masks = getappdata(obj.hGUI, obj.Type);
      if idx < 1 || idx > length(Masks)
        Mask = true(image_dims(1:3));
      else
        Mask = Masks{idx}.Mask;
      end
    end
    % --------------------------------------------------------------------
    function data = GetOriented(obj, idx, view_direction)
      data = obj.Get(idx);
      switch view_direction
        case 1, data = permute(data, [2,3,1]);
        case 2, data = permute(data, [1,3,2]);
      end
    end
    % --------------------------------------------------------------------
    function SetOriented(obj, idx, data, view_direction)
      switch view_direction
        case 1, data = permute(data, [3,1,2]);
        case 2, data = permute(data, [1,3,2]);
      end
      obj.Set(idx, data);
    end
    % --------------------------------------------------------------------
    function SetField(obj, idx, field_name, field_value)
      Masks = getappdata(obj.hGUI, obj.Type);
      if idx >= 1 && idx <= length(Masks)
        Masks{idx}.(field_name) = field_value;
        setappdata(obj.hGUI, obj.Type, Masks);
      end
    end
    % --------------------------------------------------------------------
    function field_value = GetField(obj, idx, field_name, default_value)
      Masks = getappdata(obj.hGUI, obj.Type);
      if idx >= 1 && idx <= length(Masks)
        field_value = safeget(Masks{idx}, field_name, default_value);
      end
    end
    % --------------------------------------------------------------------
    function Restore(obj)
      Masks = getappdata(obj.hGUI, obj.Type);
      MaskUndo = getappdata(obj.hGUI, 'MaskUndo');
      if isempty(MaskUndo), return; end
      MaskToRestore = MaskUndo{1};
      MaskUndo(1)=[];
      setappdata(obj.hGUI, 'MaskUndo', MaskUndo);
      
      for ii=1:length(Masks)
        if isequal(Masks{ii}.Name, MaskToRestore.Name)
          Masks{ii}.Mask = MaskToRestore.Mask;
          setappdata(obj.hGUI, obj.Type, Masks);
          break;
        end
      end
    end
    % --------------------------------------------------------------------
    % get all slice indices in the given direction
    function idx = Slices(obj, nMask, view_direction)
      Mask = obj.Get(nMask);
      idx = 1:size(Mask, view_direction);
    end
    % --------------------------------------------------------------------
    function Mask2D = Get2D(obj, idx, view_direction, sliceN)
      Mask3D = obj.Get(idx);
      if isempty(Mask3D), Mask2D = []; return; end
      Mask2D = obj.GetMask2D(Mask3D, view_direction, sliceN);
    end
    % --------------------------------------------------------------------
    function Set2D(obj, nMask, Mask, view_direction, sliceN)
      OldMask = obj.Get(nMask);
      if isempty(OldMask | ~obj.is(nMask)), return; end
      sz1 = size(OldMask);
      
      switch view_direction
        case 1, Mask = reshape(Mask, [1, sz1(2), sz1(3)]);
        case 2, Mask = reshape(Mask, [sz1(1), 1, sz1(3)]);
        case 3, Mask = reshape(Mask, [sz1(1), sz1(2), 1]);
      end
      
      if isempty(sliceN), sliceN = obj.Slices(view_direction); end
      switch view_direction
        case 1, OldMask(sliceN, :,:) = Mask;
        case 2, OldMask(:, sliceN,:) = Mask;
        case 3, OldMask(:,:, sliceN) = Mask;
      end
      obj.Set(nMask, OldMask);
    end
    % --------------------------------------------------------------------
    % A version of Set2D for use with GUI
    % apply_mode 'add', 'replace', 'erase'
    function Apply2D(obj, nMask, Mask, view_direction, sliceN, apply_mode)
      OldMask = obj.Get(nMask);
      if isempty(OldMask | ~obj.is(nMask)), return; end
      sz1 = size(OldMask);
      
      switch view_direction
        case 1, Mask = reshape(Mask, [1, sz1(2), sz1(3)]);
        case 2, Mask = reshape(Mask, [sz1(1), 1, sz1(3)]);
        case 3, Mask = reshape(Mask, [sz1(1), sz1(2), 1]);
      end
      
      if isempty(sliceN), sliceN = obj.Slices(view_direction); end
      switch apply_mode
        case 'replace'
          switch view_direction
            case 1, OldMask(sliceN, :,:) = Mask;
            case 2, OldMask(:, sliceN,:) = Mask;
            case 3, OldMask(:,:, sliceN) = Mask;
          end
        case 'add'
          switch view_direction
            case 1, OldMask(sliceN, :,:) = ...
                OldMask(sliceN, :,:) | Mask;
            case 2, OldMask(:, sliceN,:) = ...
                OldMask(:, sliceN,:) | Mask;
            case 3, OldMask(:,:, sliceN) = ...
                OldMask(:,:, sliceN) | Mask;
          end
        case 'erase'
          switch view_direction
            case 1, OldMask(sliceN, :,:) = ...
                OldMask(sliceN, :,:) & (~Mask);
            case 2, OldMask(:, sliceN,:) = ...
                OldMask(:, sliceN,:) & (~Mask);
            case 3, OldMask(:,:, sliceN) = ...
                OldMask(:,:, sliceN) & (~Mask);
          end
      end
      obj.Set(nMask, OldMask);
    end
    % --------------------------------------------------------------------
    % Math operations with single argument applied slice-wise
    function Math1s(obj, idx, idxres, view_direction, sliceN, func_name, opt)
      if ~obj.is(idx) || ~obj.is(idxres)
        warning('classMaskStorage::Math1');
        return;
      end
      Mask = obj.Get(idx);
      if isempty(sliceN), sliceN = obj.Slices(idx, view_direction); end
      
      switch func_name
        case {'ERODE', 'DILATE', 'OPEN', 'CLOSE'}
          opt.shape = safeget(opt, 'shape', 'sphere'); opt.layers = safeget(opt, 'layers', 3);
          sp = epr_strel(opt.shape, opt.layers);
        case 'LARGEST'
          nLargest = safeget(opt, 'nLargest', 1);
        otherwise
          warning(['classMaskStorage::Math1s Unknown function name ', func_name]);
      end
      
      for ii=sliceN
        switch view_direction
          case 1, MaskSlice = Mask(ii,: ,: );
          case 2, MaskSlice = Mask(: ,ii,: );
          case 3, MaskSlice = Mask(: , :,ii);
        end
        MaskSlice = squeeze(MaskSlice);
        
        switch func_name
          case 'NOT', MaskSlice = ~MaskSlice;
          case 'ERODE', MaskSlice = imerode(MaskSlice, sp);
          case 'DILATE', MaskSlice = imdilate(MaskSlice, sp);
          case 'OPEN', MaskSlice = imopen(MaskSlice, sp);
          case 'CLOSE', MaskSlice = imclose(MaskSlice, sp);
          case 'ERASE', MaskSlice(:) = 0;
          case 'LARGEST'
            [labeled_mask, nMask] = bwlabel(MaskSlice);
            if nMask > 0
              mask_volume = zeros(nMask, 1);
              for jj=1:nMask
                mask_volume(jj) = numel(find(labeled_mask == jj));
              end
              [~, max_idx] = max(mask_volume);
              MaskSlice = labeled_mask == max_idx;
            end
          case 'FILL-HOLES'
            MaskSlice = imfill(MaskSlice, 'holes');
        end
        
        switch view_direction
          case 1, Mask(ii,: ,: ) = MaskSlice;
          case 2, Mask(: ,ii,: ) = MaskSlice;
          case 3, Mask(: , :,ii) = MaskSlice;
        end
      end
      
      obj.Set(idxres, Mask);
    end
    % --------------------------------------------------------------------
    % Math operations with single argument applied image-wise
    function Math1(obj, idx, idxres, func_name, opt)
      if ~obj.is(idx) || ~obj.is(idxres)
        warning('classMaskStorage::Math1');
        return;
      end
      Mask = obj.Get(idx);
      if isempty(Mask), return; end
      
      switch func_name
        case 'COPY'
        case 'NOT', Mask = ~Mask;
        case 'ERASE', Mask(:) = 0;
        case 'SET', Mask(:) = 1;
        case 'LARGEST'
          nLargest = safeget(opt, 'nLargest', 1);
          [labeled_mask, nMask] = bwlabeln(Mask);
          mask_volume = zeros(nMask, 1);
          for ii=1:nMask
            mask_volume(ii) = numel(find(labeled_mask == ii));
          end
          if isempty(mask_volume), return; end
          [~, max_idx] = max(mask_volume);
          Mask = labeled_mask == max_idx;
        case 'FILL-HOLES'
          Mask = imfill(Mask, 'holes');
      end
      
      obj.Set(idxres, Mask);
    end
    % --------------------------------------------------------------------
    % Math operations with two arguments applied image-wise
    function Math2(obj, idx1, idx2, idxres, func_name, opt)
      if ~obj.is(idx1) || ~obj.is(idx2) || ~obj.is(idxres)
        warning('classMaskStorage::Math2');
        return;
      end
      Mask = obj.Get(idx1);
      Mask2 = obj.Get(idx2);
      
      %       switch opt
      %       end
      
      switch func_name
        case 'AND', Mask = Mask & Mask2;
        case 'OR',  Mask = Mask | Mask2;
        case 'NOTAND', Mask = (~Mask) & Mask2;
      end
      
      obj.Set(idxres, Mask);
    end
    % --------------------------------------------------------------------
    % Math operations with two arguments applied image-wise
    function Math2s(obj, idx1, idx2, idxres, view_direction, sliceN, func_name, opt)
      if ~obj.is(idx1) || ~obj.is(idx2) || ~obj.is(idxres)
        warning('classMaskStorage::Math2');
        return;
      end
      Mask = obj.Get(idx1);
      Mask2 = obj.Get(idx2);
      MaskRes = obj.Get(idxres);
      if isempty(sliceN), sliceN = obj.Slices(idx1, view_direction); end
      
      %       switch opt
      %       end
      
      for ii=sliceN
        switch view_direction
          case 1, MaskSlice = Mask(ii,: ,: ); MaskSlice2 = Mask2(ii,: ,: );
          case 2, MaskSlice = Mask(: ,ii,: ); MaskSlice2 = Mask2(: ,ii,: );
          case 3, MaskSlice = Mask(: , :,ii); MaskSlice2 = Mask2(: , :,ii);
        end
        
        switch func_name
          case 'AND', MaskSlice = MaskSlice & MaskSlice2;
          case 'OR',  MaskSlice = MaskSlice | MaskSlice2;
          case 'NOTAND', MaskSlice = ((~MaskSlice) & MaskSlice2) | ((~MaskSlice2) & MaskSlice);
        end
        
        switch view_direction
          case 1, MaskRes(ii,: ,: ) = MaskSlice;
          case 2, MaskRes(: ,ii,: ) = MaskSlice;
          case 3, MaskRes(: , :,ii) = MaskSlice;
        end
      end
      
      obj.Set(idxres, MaskRes);
    end
    % --------------------------------------------------------------------
    % Math operations with two arguments applied image-wise
    function val = Metric2(obj, idx1, idx2, func_name, opt)
      if ~obj.is(idx1) || ~obj.is(idx2)
        warning('classMaskStorage::Metric2');
        return;
      end
      Mask = obj.Get(idx1);
      Mask2 = obj.Get(idx2);
      
      switch func_name
        case 'DICE', val = compare_dice(Mask,Mask2);
        case 'HAUSDORFF'
          [val.hd, val.D] = HausdorffDist(Mask,Mask2);
      end
    end
  end
  methods (Static)
    % --------------------------------------------------------------------
    function data = Oriented(data, view_direction)
      switch view_direction
        case 1, data = permute(data, [2,3,1]);
        case 2, data = permute(data, [1,3,2]);
      end
    end
    % --------------------------------------------------------------------
    function Mask2D = GetMask2D(Mask3D, view_direction, sliceN)
      if isempty(Mask3D), Mask2D = []; return; end
      
      switch view_direction
        case 1, Mask2D = squeeze(Mask3D(sliceN, :, :));
        case 2, Mask2D = squeeze(Mask3D(:, sliceN, :));
        case 3, Mask2D = Mask3D(:, :, sliceN);
      end
    end
    % --------------------------------------------------------------------
    function [yx, zx, yz]=epr_getslice3D(data, proj, idx_x, idx_y, idx_z)
      
      if ~exist('idx_x', 'var'), idx_x = 1:size(data,2); end
      if ~exist('idx_y', 'var'), idx_y = 1:size(data,1); end
      if ~exist('idx_z', 'var'), idx_z = 1:size(data,3); end
      
      if isempty(proj)
        proj = fix(0.5*size(data));
      end
      
      yx = squeeze(data(idx_x,idx_y,proj(3)));
      zx = squeeze(data(idx_x,proj(1),idx_z));
      yz = squeeze(data(proj(2),idx_y,idx_z))';
    end
    % --------------------------------------------------------------------
    function res = GetFunctionArguments(func_name)
      res = {};
      switch func_name
        case 'Math1s', res =  {'COPY','ERODE','DILATE','OPEN','CLOSE','ERASE','LARGEST','FILL-HOLES','NOT'};
        case 'Math1', res = {'COPY','NOT','ERASE','SET','LARGEST','FILL-HOLES'};
        case 'Math2', res =  {'AND','OR','NOTAND'};
        case 'Math2s', res =  {'AND','OR','NOTAND'};
        case 'Metric2', res =  {'DICE','HAUSDORFF'};
      end
    end
  end
end