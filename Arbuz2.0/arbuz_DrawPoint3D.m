function arbuz_DrawPoint3D(ax_hh, img, AA)

color_structure = safeget(img, 'Color', []);

Color     = safeget(color_structure, 'Color', 'blue');
LineWidth = safeget(color_structure, 'LineWidth', 0.5);
% FaceColor = safeget(color_structure, 'FaceColor', 'blue');
% EdgeColor = safeget(color_structure, 'EdgeColor', 'blue');
% FaceAlpha = safeget(color_structure, 'FaceAlpha', 1);
% FaceVertexCData = safeget(color_structure, 'FaceVertexCData', 'blue');
% FaceLighting = safeget(color_structure, 'FaceLighting', 'flat');
% BackFaceLighting = safeget(color_structure, 'BackFaceLighting', 'unlit');

xyz = img.data;
h = [];
if size(xyz,1) == 1
  r(1,:) = xyz + [25,0,0];
  r(2,:) = xyz + [-25,0,0];
  r = htransform_vectors(AA,r);
  h(end+1) = plot3(r(:,1),r(:,2),r(:,3), 'Parent', ax_hh);
  r(1,:) = xyz + [0,-25,0];
  r(2,:) = xyz + [0,25,0];
  r = htransform_vectors(AA,r);
  h(end+1) = plot3(r(:,1),r(:,2),r(:,3), 'Parent', ax_hh);
  r(1,:) = xyz + [0,0,-25];
  r(2,:) = xyz + [0,0,25];
  r = htransform_vectors(AA,r);
  h(end+1) = plot3(r(:,1),r(:,2),r(:,3), 'Parent', ax_hh);
else
  r = htransform_vectors(AA,xyz);
  if size(xyz,1) == 2
    dr = diff(r,1,1);
    r(1,:) = r(1,:) - dr*0.1;
    r(2,:) = r(2,:) + dr*0.1;
  end
  h(end+1) = plot3(r(:,1),r(:,2),r(:,3), 'Parent', ax_hh);
end
set(h, 'LineWidth', LineWidth, 'Color', Color);
