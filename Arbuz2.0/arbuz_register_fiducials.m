function [A, minERR] = arbuz_register_fiducials(rotating_fiducials, static_fiducials, opts)

fiducial_number = safeget(opts, 'fiducial_number', 4);
if length(rotating_fiducials) < fiducial_number 
  fprintf(2, 'arbuz_register_fiducials: %i rotating fiducials supplied.\n', length(rotating_fiducials));
  return; 
end
if length(static_fiducials) < fiducial_number 
  fprintf(2, 'arbuz_register_fiducials: %i static fiducials supplied.\n', length(static_fiducials));
  return; 
end
if length(rotating_fiducials) ~= fiducial_number, fprintf('arbuz_register_fiducials: warning %i fiducials supplied.\n', length(rotating_fiducials)); end
if length(static_fiducials) ~= fiducial_number, fprintf('arbuz_register_fiducials: warning %i fiducials supplied.\n', length(rotating_fiducials)); end

rotating_fiducials = rotating_fiducials(1:fiducial_number);
static_fiducials = static_fiducials(1:fiducial_number);

% figure out fiducials order
Aknown   = safeget(opts, 'Aknown', eye(4));

for ii=1:length(rotating_fiducials)
  rotating_fiducials{ii}.pts = htransform_vectors(rotating_fiducials{ii}.Anative*Aknown, rotating_fiducials{ii}.data);
  [rotating_fiducials{ii}.a, rotating_fiducials{ii}.r] = get_best_line3D(rotating_fiducials{ii}.pts);
end
rf_order = get_fiducial_index(rotating_fiducials);
for ii=1:length(static_fiducials)
  static_fiducials{ii}.pts = htransform_vectors(static_fiducials{ii}.Anative, static_fiducials{ii}.data);
  [static_fiducials{ii}.a, static_fiducials{ii}.r] = get_best_line3D(static_fiducials{ii}.pts);
end
sf_order = get_fiducial_index(static_fiducials);

rf_idx = rf_order(sf_order);

Aknown   = safeget(opts, 'Aknown', eye(4));
MaxIter  = safeget(opts, 'maximum_fit_iterations', 1e5);

npoints = 4;
fit_array = zeros(length(rotating_fiducials)*npoints,9);
for ii=1:length(rotating_fiducials)
  rf = htransform_vectors(rotating_fiducials{rf_idx(ii)}.Anative*Aknown, rotating_fiducials{rf_idx(ii)}.data);
  sf = htransform_vectors(static_fiducials{ii}.Anative, static_fiducials{ii}.data);
  [a, r] = get_best_line3D(sf);
  drf = diff(rf, 1, 1);
  fit_array((ii-1)*npoints+1, :) = [rf(1,:), a, r];
  fit_array((ii-1)*npoints+2, :) = [rf(2,:), a, r];
  fit_array((ii-1)*npoints+3, :) = [rf(1,:)-drf*0.25, a, r];
  fit_array((ii-1)*npoints+4, :) = [rf(2,:)+drf*0.25, a, r];
end

options = optimset('TolFun',1e-12,'TolX',1e-12,'MaxFunEvals',MaxIter,'Display','none','MaxIter',MaxIter);
RES = fminsearch(@fit_func1, [0,0,0,0,0,0], options, fit_array);
startERR = fit_func1([0,0,0,0,0,0], fit_array);
minERR = fit_func1(RES, fit_array);

fprintf('arbuz_register_fiducials: error = %4.3f (%4.3f)\n', minERR, startERR);

Afit = hmatrix_rotate_euler(RES(4:6)) * hmatrix_translate(RES(1:3));
A = Aknown * Afit;
% A = eye(4);
%% report
figN  = safeget(opts, 'figure', 1100);
fname = safeget(opts, 'figure_filename', '');
 
if figN > -1 && ~isempty(fname)
  figure(1000+figN); clf; hold on;
  color = {'r-', 'b-', 'k-', 'g-', 'm-'};
  for ii=1:length(static_fiducials)
    sf = htransform_vectors(static_fiducials{ii}.Anative, static_fiducials{ii}.data);
    plot3(sf(:,1), sf(:,2), sf(:,3),color{ii})
  end
  for ii=1:length(rotating_fiducials)
    for jj=1:npoints
      rf = htransform_vectors(Afit, fit_array((ii-1)*npoints+jj,1:3));
      plot3(rf(:,1), rf(:,2), rf(:,3),['o',color{ii}(1)])
    end
  end
  box on; xlabel('x'); ylabel('y'); zlabel('z'); grid on
  title('1: red 2: blue 3: black 4: green');
  epr_mkdir(fileparts([fname,num2str(figN),'.fig']));
  savefig(1000+figN , [fname,num2str(figN),'.fig'] , 'compact' )
  delete(1000+figN);
end
% --------------------------------------------------------------------
function error = fit_func1(x, fit_array)
A = hmatrix_rotate_euler(x(4:6)) * hmatrix_translate(x(1:3));
dist_array = fit_array;
dist_array(:,1:3) = htransform_vectors(A, fit_array(:,1:3));
error = 0;
for ii = 1:size(fit_array, 1)
  error = error + point_to_line_distance(dist_array(ii,:)).^2;
end
error = sqrt(error);
% disp(error);

% --------------------------------------------------------------------
function error = fit_func2(x, fit_array)
A = hmatrix_translate(x(1:3));
dist_array = fit_array;
dist_array(:,1:3) = htransform_vectors(A, fit_array(:,1:3));
error = 0;
for ii = 1:size(fit_array, 1)
  error = error + point_to_line_distance(dist_array(ii,:)).^2;
end
error = sqrt(error);
% disp(error);

% --------------------------------------------------------------------
function error = fit_func3(x, fit_array)
A = hmatrix_rotate_euler(x(1:3));
dist_array = fit_array;
dist_array(:,1:3) = htransform_vectors(A, fit_array(:,1:3));
error = 0;
for ii = 1:size(fit_array, 1)
  error = error + point_to_line_distance(dist_array(ii,:)).^2;
end
error = sqrt(error);
% disp(error);

% --------------------------------------------------------------------
function [r0, a] = get_best_line3D(points)

N = size(points, 1);
r0 = mean(points);
xyz = points - repmat(r0, [N, 1]);
[~,~,V]=svd(xyz,0); 
a = V(1:3,1)';   

% --------------------------------------------------------------------
function res = point_to_line_distance(FA) 

res = norm(cross(FA(1:3) - FA(4:6), FA(7:9))) / norm(FA(7:9));

% --------------------------------------------------------------------
% find fiducials order [slanted, nearest, .. , farest]
function res = get_fiducial_index(fid) 

nfid = length(fid);
fit_array = zeros(nfid,9);
for ii=1:length(fid)
  for jj=1:length(fid)
    pts = fid{ii}.pts;
    a = fid{jj}.a;
    r = fid{jj}.r;
    drf = diff(pts, 1, 1);

    fit_array(1, :) = [pts(1,:), a, r];
    fit_array(2, :) = [pts(2,:), a, r];
    fit_array(3, :) = [pts(1,:)-drf*0.25, a, r];
    fit_array(4, :) = [pts(2,:)+drf*0.25, a, r];
    fid{ii}.err(jj) = fit_func1([0,0,0,0,0,0], fit_array);
  end
end
all_together = zeros(nfid,nfid-1);
for ii=1:length(fid)
  all_together(ii,:) = fid{ii}.err([1:ii-1,ii+1:end]);
end
all_together_sort = sort(all_together(:));
all_together_sort = mean(reshape(all_together_sort, [2,(nfid-1)*2]));

% get fid 1 (slanted)
[~,fid1] = max(mean(all_together, 2));
% get second one
fid3_4 = find(min(all_together, [], 2) < all_together_sort(1)*1.1);
global_idx = 1:4;
fid2 = find(global_idx ~= fid1 & global_idx ~= fid3_4(1) & global_idx ~= fid3_4(2));
fid3_4 = fid3_4(fid3_4 ~= fid2);
% get other two
[~,idx] = max(mean(all_together(fid3_4,:), 2));
fid4 = fid3_4(idx);
fid3 = fid3_4([1:idx-1,idx+1:end]);

res = [fid1, fid2, fid3, fid4];
