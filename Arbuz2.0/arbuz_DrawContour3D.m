% --------------------------------------------------------------------
function arbuz_DrawContour3D(ax_hh, img, AA)

color_structure = safeget(img, 'Color', []);

Color = safeget(color_structure, 'Color', 'blue');
LineWidth = safeget(color_structure, 'LineWidth', 0.5);
% FaceColor = safeget(color, 'FaceColor', 'blue');
% EdgeColor = safeget(color, 'EdgeColor', 'blue');
% FaceAlpha = safeget(color, 'FaceAlpha', 1);
% FaceVertexCData = safeget(color, 'FaceVertexCData', 'blue');
% FaceLighting = safeget(colorr, 'FaceLighting', 'flat');
% BackFaceLighting = safeget(color, 'BackFaceLighting', 'unlit');

ii = 1;
hold(ax_hh, 'on');
while ii < size(img.data,1)
  p = img.data(ii,2); r1 = img.data(ii+1:ii+p,:); ii = ii+1+p;
  r1 = htransform_vectors(AA,r1);
  h = plot3(r1(:,1),r1(:,2),r1(:,3), 'Parent', ax_hh);
  set(h, 'Color', Color, 'LineWidth', LineWidth);
end