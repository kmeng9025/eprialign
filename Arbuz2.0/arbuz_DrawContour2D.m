function arbuz_DrawContour2D(ax_hh, img, AA)

color_structure = safeget(img, 'Color', []);

Color = safeget(color_structure, 'Color', 'blue');
LineWidth = safeget(color_structure, 'LineWidth', 0.5);
% FaceColor = safeget(color_structure, 'FaceColor', 'blue');
% EdgeColor = safeget(color_structure, 'EdgeColor', 'blue');
% FaceAlpha = safeget(color_structure, 'FaceAlpha', 1);
% FaceVertexCData = safeget(color_structure, 'FaceVertexCData', 'blue');
% FaceLighting = safeget(color_structure, 'FaceLighting', 'flat');
% BackFaceLighting = safeget(color_structure, 'BackFaceLighting', 'unlit');

ii = 1;
h = [];
hold(ax_hh, 'on');
while ii < size(img.data,1)
  p = img.data(ii,2); r1 = img.data(ii+1:ii+p,:); ii = ii+1+p;
  r1 = htransform_vectors(AA,r1);
  h(end+1) = plot(r1(:,1),r1(:,2),'Parent', ax_hh);
end
if isfield(img, 'LineWidth')
  set(h, 'LineWidth', img.LineWidth);
end
set(h, 'Color', Color, 'LineWidth', LineWidth);