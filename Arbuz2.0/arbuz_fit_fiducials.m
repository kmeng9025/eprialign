function [res] = arbuz_fit_fiducials(image, opts)

fiducial_number = safeget(opts, 'fiducial_number', 4);

% clear out excessive fiducials
CC = bwconncomp(image);
if CC.NumObjects > fiducial_number
  items = cellfun(@(x) numel(x), CC.PixelIdxList);
  [~,idx] = sort(items, 'descend');
  for ii=fiducial_number+1:length(items)
    image(CC.PixelIdxList{idx(ii)}) = false;
  end
  CC = bwconncomp(image);
end

sz = size(image);

A = safeget(opts, 'A', eye(4));
x = 1:sz(2);
y = 1:sz(1);
z = 1:sz(3);

[X,Y,Z] = meshgrid(x,y,z);

res = cell(CC.NumObjects, 1);

for ii=1:CC.NumObjects
  % find best line through the object
  idx = CC.PixelIdxList{ii};
  points = [X(idx),Y(idx),Z(idx)];
  points = htransform_vectors(A, points);
  
  N = size(points, 1);
  res{ii}.r0 = mean(points);
  xyz = points - repmat(res{ii}.r0, [N, 1]);
  [~,~,V]=svd(xyz,0);
  res{ii}.a = V(1:3,1)';
  range = [min(X(idx)), min(Y(idx)), min(Z(idx)); max(X(idx)), max(Y(idx)), max(Z(idx))];
  range = htransform_vectors(A, range);
  
  [~, res{ii}.max_range] = max(diff(range, 1, 1));
  prange = range(:, res{ii}.max_range)';
  prange = (prange - res{ii}.r0(res{ii}.max_range))./res{ii}.a(res{ii}.max_range);
  
  ends(1,:) = res{ii}.r0+res{ii}.a*prange(1);
  ends(2,:) = res{ii}.r0+res{ii}.a*prange(2);

  res{ii}.ends = htransform_vectors(inv(A), ends);
  res{ii}.range = htransform_vectors(inv(A), range)';

%   res = 0;
%   for jj=1:handles.obj{ii}.N
%     res = res + (norm(cross(points(jj,:) - r0, a)) / norm(a))^2;
%   end
%   handles.obj{ii}.std = sqrt(res/handles.obj{ii}.N);
%   
end

%%

%% report figure
figN  = safeget(opts, 'figure', 1);
fname = safeget(opts, 'figure_filename', '');
if figN > -1 && ~isempty(fname)
    color = {'k', 'b', 'm', 'r', 'g', 'c', 'y', 'k', 'b', 'm', 'r', 'g', 'c'};
    figure(1000+figN); clf; hold on
    
    for ii=1:length(res)
        idx = CC.PixelIdxList{ii};
        points = [X(idx),Y(idx),Z(idx)];
        points = htransform_vectors(A, points);
        plot3(points(:,1), points(:,2), points(:,3), 'Color', color{ii}, 'Marker', '.', 'lineStyle', 'none')
        p1 = res{ii}.ends;
        p1 = htransform_vectors(A, p1);
        plot3(p1(:,1),p1(:,2),p1(:,3), 'Color', color{ii})
    end
    axis image; box on; grid on
    
    %set(figN, 'Position', get(0, 'Screensize'));
%     saveas(figN, [fname,'1.png']);
    epr_mkdir(fileparts([fname,num2str(figN),'.fig']));
    savefig( 1000+figN , [fname,num2str(figN),'.fig'] , 'compact' ) 
%     delete( 1000+figN );
end

