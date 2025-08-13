function arbuz_CopyAllTransformations(hGUI, image_src, image_dest, first_transformation)

% load the project
prj = getappdata(hGUI, 'project');

image_src = arbuz_FindImage(hGUI, image_src, '', '', {});
image_dest = arbuz_FindImage(hGUI, image_dest, '', '', {});

for ii=first_transformation:length(prj.Transformations)
  % clear all transformation from the destination image
  Matrices = {};
  for jj=1:length(prj.Transformations{ii}.Matrices)
    if ~strcmp(prj.Transformations{ii}.Matrices{jj}.Image, image_dest{1}.Image)
      Matrices{end+1} = prj.Transformations{ii}.Matrices{jj};
    end
  end
  
  % add
  for jj=1:length(Matrices)
    if strcmp(Matrices{jj}.Image, image_src{1}.Image)
      new_transformation.Image = image_dest{1}.Image;
      new_transformation.A = Matrices{jj}.A;
      Matrices{end+1} = new_transformation;
      break;
    end
  end
  
  prj.Transformations{ii}.Matrices = Matrices;
end

% save the project
setappdata(hGUI, 'project', prj);

arbuz_SetActiveTransformation(hGUI);
