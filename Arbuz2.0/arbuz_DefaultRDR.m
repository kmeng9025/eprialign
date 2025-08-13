function arbuz_DefaultRDR(hGUI, options)

field_names =   {'Name', 'FullName', 'FileName', 'data', 'Mask', 'Color', 'box', 'Anative', 'isLoaded', 'SlaveList'};
img = arbuz_FindImage(hGUI, {options.image}, '', '', field_names);
if isempty(img), return; end
img = img{1};

% default viewer for image types
h = [];
switch img.ImageType
  case '2D'
    %       h = imtool(img.data);
    h = figure;
    image(img.data);
  case 'XYZ'
    im = arbuz_FindImage(hGUI, img, '', '', {'Ashow'});
    ddd = htransform_vectors(im{1}.Ashow,img.data);
    fprintf('\n');
    for ii=1:size(img.data,1)
      arbuz_ShowMessage(hGUI, sprintf('%s:XYZ(origin) = %f %f %f', img.Name, img.data(ii,:)));
      arbuz_ShowMessage(hGUI, sprintf('%s:XYZ(show) = %f %f %f', img.Name, ddd(ii,1:3)))
    end
  case 'CONTOUR'
    h = figure;
    arbuz_DrawContour2D(gca, img, eye(4));
  case {'3DEPRI','FITRESULT', 'PO2_pEPRI','AMP_pEPRI','3DMASK','MRI','RAW', 'AMIRA3D', 'DICOM3D','IDL','BIN','JIVATDMS','CEST_ML'}
    data.(matlab.lang.makeValidName(img.Name)) = img.data;
    data.Masks = {};
    if ~isempty(img.Mask)
      data.Mask = img.Mask;
      im_mask = arbuz_FindImage(hGUI, img.SlaveList, 'ImageType', '3DMASK', {'Name', 'data'});
      
      for ii=1:length(im_mask)
          data.Masks{end+1} = struct('Mask', im_mask{ii}.data, 'Name', im_mask{ii}.Name);
      end
    elseif ~isempty(img.SlaveList)
      data.Mask = true(size(img.data));
      im_mask = arbuz_FindImage(hGUI, img.SlaveList, 'ImageType', '3DMASK', {'Name', 'data'});
      
      for ii=1:length(im_mask)
          data.Masks{end+1} = struct('Mask', im_mask{ii}.data, 'Name', im_mask{ii}.Name);
      end
    end
    ibGUI(data);
  case {'pEPROI'}
    data.pO2 = img.data;
    slave_images = arbuz_FindImage(hGUI, img.SlaveList, '', '', {'Name', 'data'});
    for ii=1:length(slave_images)
          data.(slave_images{ii}.Name) = slave_images{ii}.data;
    end
    ibGUI(data);
  case '3DSURFACE'
    h = figure;
    arbuz_DrawSurface3D(gca, img, eye(4));
    
    cameratoolbar(h,'NoReset')
    cameratoolbar(h,'Show')
    cameratoolbar(h,'SetMode','orbit')
    cameratoolbar(h,'SetCoordSys','y')
    drawnow
    grid on
end
if ~isempty(h)
  set(h, 'Name', epr_ShortFileName(img.FullName, 60));
  axis image;
end
