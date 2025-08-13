function h = arbuz_DrawPoint2D(ax_hh, img, AA)

color_structure = safeget(img, 'Color', []);

Color = safeget(color_structure, 'Color', 'blue');
LineWidth = safeget(color_structure, 'LineWidth', 0.5);
% FaceColor = safeget(color, 'FaceColor', 'blue');
% EdgeColor = safeget(color, 'EdgeColor', 'blue');
% FaceAlpha = safeget(color, 'FaceAlpha', 1);
% FaceVertexCData = safeget(color, 'FaceVertexCData', 'blue');
% FaceLighting = safeget(colorr, 'FaceLighting', 'flat');
% BackFaceLighting = safeget(color, 'BackFaceLighting', 'unlit');
PrintLabel = safeget(color_structure, 'PrintLabel', 0);

xyz     = img.data;
tick_sz = 50;
h = [];
r(1,:) = xyz + [tick_sz/2,0,0];
r(2,:) = xyz + [-tick_sz/2,0,0];
r = htransform_vectors(AA,r);
h(end+1) = plot(r(:,1),r(:,2), 'Parent', ax_hh);
r(1,:) = xyz + [0,-tick_sz/2,0];
r(2,:) = xyz + [0,tick_sz/2,0];
r = htransform_vectors(AA,r);
h(end+1) = plot(r(:,1),r(:,2), 'Parent', ax_hh);
set(h, 'Color', Color, 'LineWidth', LineWidth);

if PrintLabel
  r = htransform_vectors(AA,xyz+[tick_sz/2,-tick_sz/2,0]);
  text(r(1), r(2), img.Name, ...
    'Color', [1,1,1], 'BackgroundColor', [0,0,0], ...
    'VerticalAlignment', 'bottom', 'Tag', 'Anchors');
end
