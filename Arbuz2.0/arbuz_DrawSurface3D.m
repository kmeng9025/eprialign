function arbuz_DrawSurface3D(ax_hh, img, AA)

color_structure = safeget(img, 'Color', []);

% Color = safeget(color_structure, 'Color', 'blue');
FaceColor = safeget(color_structure, 'FaceColor', 'blue');
EdgeColor = safeget(color_structure, 'EdgeColor', 'blue');
FaceAlpha = safeget(color_structure, 'FaceAlpha', 1);
% FaceVertexCData = safeget(color_structure, 'FaceVertexCData', 'blue');
FaceLighting = safeget(color_structure, 'FaceLighting', 'flat');
BackFaceLighting = safeget(color_structure, 'BackFaceLighting', 'unlit');

if isempty(img.data.vert)
  warning(['Mask is empty: ',img.Image,':',img.Slave, '.']);
  return;
end

vert = htransform_vectors(AA, img.data.vert);

patch('Parent', ax_hh, ...
  'faces',img.data.face,'vertices',vert, ...
  'facecolor', FaceColor, 'facelighting',FaceLighting, ...
  'edgecolor', EdgeColor, 'FaceAlpha',FaceAlpha, ...
  'BackFaceLighting',BackFaceLighting ...
  );

%   'facevertexCdata',FaceVertexCData, ...



