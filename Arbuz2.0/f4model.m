function [A, A2, rot_f4r, rot_f4r1, ret] = f4model(fiducials, resolution, opt)
ret = [];
n = 6;

figN  = safeget(opt, 'figure', 4000);
fname = safeget(opt, 'figure_filename', '');

stages = safeget(opt, 'stages', [0.1, 0.2, 0.4]);
erode_layers = safeget(opt, 'erode_layers', [0,0,0]);
guess = safeget(opt, 'guess', zeros(1,n));
extentz = safeget(opt, 'extentz', [-100,100]);

defaults = opt.defaults;

usepause = safeget(opt, 'pause', false);

start_choice = 1;
start_parameters{1}.A = eye(4);
start_parameters{2}.A = hmatrix_rotate_y(180);
start_parameters{3}.A = hmatrix_scale([1,1,-1])*hmatrix_rotate_y(90)*hmatrix_rotate_x(180);

if contains(defaults, 'MRI')
    % first dimension flip
    dim2 = squeeze(sum(sum(fiducials, 1), 3)); dim2 = dim2(:)'; dim2(dim2 < 0.1*max(dim2)) = 0;
    [~,locs] = findpeaks(dim2,'MinPeakDistance',6);

    if numel(locs) == 4
        cm = mean(locs);

        h1 = squeeze(sum(sum(fiducials(:,1:floor(cm),:), 2), 3)); h1 = h1(:)';
        h2 = squeeze(sum(sum(fiducials(:,floor(cm):end,:), 2), 3)); h2 = h2(:)';
        h1(h1 < 0.05*max(h1)) = 0;
        h2(h2 < 0.05*max(h2)) = 0;
        [h1p] = findpeaks(h1,'MinPeakDistance',5);
        [h2p] = findpeaks(h2,'MinPeakDistance',5);

        % diagonal is in h1
        if mean(h1p) < 0.5*max(h1p)
            defaults = 'MRI180';
            % diagonal is in h2
        elseif mean(h2p) < 0.5*max(h2p)
            defaults = 'MRI';
        end

        if 1==0
            figure(2000);clf
            plot(1:length(dim2), dim2); hold on
            dim1 = squeeze(sum(sum(fiducials, 2), 3)); dim1 = dim1(:)';
            plot(1:length(dim1), dim1); hold on
            dim3 = squeeze(sum(sum(fiducials, 1), 2)); dim3 = dim3(:)';
            plot(1:length(dim3), dim3); hold on
            plot(locs, dim2(locs), '*')
            plot([cm,cm], [0, max(h1)], ':')
            legend({'dim2', 'dim1', 'dim3'})
        end
    elseif numel(locs) < 4
        fprintf('MIRRORS DETECTION: Peaks detected (%i)\n', numel(locs));
        [~,locs] = findpeaks(dim2,'MinPeakDistance',2);

        if numel(locs) == 4
            cm = mean(locs);

            h1 = squeeze(sum(sum(fiducials(:,1:floor(cm),:), 2), 3)); h1 = h1(:)';
            h2 = squeeze(sum(sum(fiducials(:,floor(cm):end,:), 2), 3)); h2 = h2(:)';
            h1(h1 < 0.05*max(h1)) = 0;
            h2(h2 < 0.05*max(h2)) = 0;
            [h1p] = findpeaks(h1,'MinPeakDistance',5);
            [h2p] = findpeaks(h2,'MinPeakDistance',5);

            % diagonal is in h1
            if mean(h1p) < 0.5*max(h1p)
                defaults = 'MRI180';
                % diagonal is in h2
            elseif mean(h2p) < 0.5*max(h2p)
                defaults = 'MRI';
            end
        else
        fprintf('MIRRORS DETECTION: Peak based determination (%i) did not work\n', numel(locs));
        figure(2000);clf
        plot(1:length(dim2), dim2); hold on
        dim1 = squeeze(sum(sum(fiducials, 2), 3)); dim1 = dim1(:)';
        plot(1:length(dim1), dim1); hold on
        dim3 = squeeze(sum(sum(fiducials, 1), 2)); dim3 = dim3(:)';
        plot(1:length(dim3), dim3); hold on
        plot(locs, dim2(locs), '*')
        legend({'dim2', 'dim1', 'dim3'})
        end
    else
        fprintf('MIRRORS DETECTION: Peak based determination (%i) did not work\n', numel(locs));
        figure(2000);clf
        plot(1:length(dim2), dim2); hold on
        dim1 = squeeze(sum(sum(fiducials, 2), 3)); dim1 = dim1(:)';
        plot(1:length(dim1), dim1); hold on
        dim3 = squeeze(sum(sum(fiducials, 1), 2)); dim3 = dim3(:)';
        plot(1:length(dim3), dim3); hold on
        plot(locs, dim2(locs), '*')
        legend({'dim2', 'dim1', 'dim3'})
    end

end

switch defaults
    case 'CT'
        A1 = hmatrix_scale([-1,1,1]);
    case 'MRI'
        A1 = eye(4);
    case 'MRI180'
        start_choice = 2;
        A1 = start_parameters{start_choice}.A;
    case 'EPR'
        start_choice = 3;
        A1 = start_parameters{start_choice}.A;
    otherwise
        A1 = eye(4);
end

nSize = size(fiducials);

% voxel to coordinates
if all(size(resolution) == [4,4])
    A2 = resolution;
else
    ImageSize = nSize .* resolution(1:3);
    A2 = hmatrix_translate(-nSize/2) * hmatrix_scale(ImageSize./nSize);
end
xv = 1:nSize(1);
yv = 1:nSize(2);
zv = 1:nSize(3);
[XV,YV,ZV]=meshgrid(yv,xv,zv);

% x=linspace(-ImageSize(1)/2,ImageSize(1)/2,nSize(1));
% y=linspace(-ImageSize(2)/2,ImageSize(2)/2,nSize(2));
% z=linspace(-ImageSize(3)/2,ImageSize(3)/2,nSize(3));
% [X,Y,Z]=meshgrid(y,x,z);
% XYZ = [X(fidpoints),Y(fidpoints),Z(fidpoints)];

tic
x = zeros(1,n);

ErrTH = safeget(opt, 'fit_threshold', 10000);

for ii=1:length(stages)
    if erode_layers(ii) > 0
        se = epr_strel('sphere',erode_layers(ii));
        fiducials2 = imerode(fiducials, se);
    else
        fiducials2 = fiducials;
    end
    fidpoints = find(fiducials2);

    XYZV = [XV(fidpoints),YV(fidpoints),ZV(fidpoints)];
    XYZ = htransform_vectors(A2,XYZV);

    % clear outlyers
    if ii > 1
        % model specific clear around fiducials and remove ends of fiducials
        [rot_f4r, rot_f4r1] = get_model6(x, A1);
        v1 = geom_point2line_threshold(XYZ, rot_f4r(1,:), rot_f4r1(1,:), 2.3);
        v2 = geom_point2line_threshold(XYZ, rot_f4r(2,:), rot_f4r1(2,:), 2.3);
        v3 = geom_point2line_threshold(XYZ, rot_f4r(3,:), rot_f4r1(3,:), 2.3);
        v4 = geom_point2line_threshold(XYZ, rot_f4r(4,:), rot_f4r1(4,:), 2.3);
        XYZ1 = XYZ(v1|v2|v3|v4,:);

        XYZ1 = XYZ1(XYZ1(:,3)> x(3)+extentz(1) & XYZ1(:,3) < x(3)+extentz(2), :);
    else
        XYZ1 = XYZ;
    end

    a = randn(size(XYZ1, 1),1);
    numelements = round(stages(ii)*length(a)); % CT 0.01
    rand_idx = randperm(length(a),numelements);
    datasubset = XYZ1(rand_idx,:);

    if usepause
        [rot_f4r, rot_f4r1, A] = get_model6(x, A1);
        figure(figN); clf; hold on
        d = rot_f4r1 - rot_f4r;
        rot_f4r = rot_f4r - 2*d;
        rot_f4r1 = rot_f4r1 + 6*d;
        for ll=1:4
            plot3([rot_f4r(ll,1), rot_f4r1(ll,1)], [rot_f4r(ll,2), rot_f4r1(ll,2)], [rot_f4r(ll,3), rot_f4r1(ll,3)], '.-', ...
                'MarkerSize', 22,'linewidth',2)
        end
        for ll=1:size(datasubset,1)
            plot3(datasubset(ll,1),datasubset(ll,2), datasubset(ll,3),'.')
        end
        axis equal
        disp(cost_func(x, datasubset, A1))
        pause
    end

    x = fminsearch(@cost_func, x, [], datasubset, A1);

    if ii == 1 && cost_func(x, datasubset, A1) > ErrTH
        disp('Potential error detected. Starting search.');
        start_parameters{start_choice}.x = x;
        cost = zeros(1,length(start_parameters));
        cost(start_choice) = cost_func(x, datasubset, A1);
        for kk = 1:length(start_parameters)
            if kk ~= start_choice
                start_parameters{kk}.x = fminsearch(@cost_func, zeros(1,n), [], datasubset, start_parameters{kk}.A);
                cost(kk) = cost_func(start_parameters{kk}.x, datasubset, start_parameters{kk}.A);
            end
        end
        [~,start_choice] = min(cost);
        A1 = start_parameters{start_choice}.A;
        x = start_parameters{start_choice}.x;
    end

    if numel(x) < 6, x(6) = 0; end
    ret.cost(ii) = cost_func(x, datasubset, A1);
    fprintf('f4model: Iteration %i [%.3f %.3f %.3f %.3f %.3f] CF=%.4f (%i)\n', ii, x(1), x(2), x(3), x(4), x(5), cost_func(x, datasubset, A1),numelements)
    x = x(1:n);
end
x = x(1:n);

toc;

[rot_f4r, rot_f4r1, A] = get_model6(x, A1);

% return

%
if figN > -1 && ~isempty(fname)

    [rot_f4r, rot_f4r1, A] = get_model6(x, A1);
    figure(figN); clf; hold on
    d = rot_f4r1 - rot_f4r;
    rot_f4r = rot_f4r - 2*d;
    rot_f4r1 = rot_f4r1 + 6*d;
    for ii=1:4
        plot3([rot_f4r(ii,1), rot_f4r1(ii,1)], [rot_f4r(ii,2), rot_f4r1(ii,2)], [rot_f4r(ii,3), rot_f4r1(ii,3)], '.-', ...
            'MarkerSize', 22,'linewidth',2)
    end
    for ii=1:size(datasubset,1)
        plot3(datasubset(ii,1),datasubset(ii,2), datasubset(ii,3),'.')
    end
    axis equal

    fname = [fname, num2str(figN),'A.fig'];
    epr_mkdir(fileparts(fname));
    saveas(figN, fname);
    delete(figN);
end
%

return

function [rot_f4r, rot_f4r1, A] = get_model6(x, A1)
%

fa  = 20*pi/180;

f1a = 22.5*pi/180; % 22.5*pi/180
f2a = 0*pi/180;
f3a = -45*pi/180; % -45*pi/180;
fr  = 8.15; %  8.15;
fr4 = 7.7; % 7.7

allf4a = [...
    0    0       1;    ...
    0    0       1;    ...
    0    0       1;    ...
    0 -sin(fa)  cos(fa)  ...
    ];

allf4r = [...
    -fr*sin(f1a)  fr*cos(f1a)     0;   ...
    -fr*sin(f2a)  fr*cos(f2a)     0;   ...
    -fr*sin(f3a)  fr*cos(f3a)     0;    ...
    fr4    2     0    ...
    ];

rot_f4r   = allf4r;
rot_f4r1  = rot_f4r + allf4a;

rot_f4r(:,4) = 1;
rot_f4r1(:,4) = 1;

A = A1 * hmatrix_rotate_euler(x(4:6)) * hmatrix_translate(x(1:3));

rot_f4r(1,:) = rot_f4r(1,:) * A;
rot_f4r(2,:) = rot_f4r(2,:) * A;
rot_f4r(3,:) = rot_f4r(3,:) * A;
rot_f4r(4,:) = rot_f4r(4,:) * A;
rot_f4r = rot_f4r(:,1:3);

rot_f4r1(1,:) = rot_f4r1(1,:) * A;
rot_f4r1(2,:) = rot_f4r1(2,:) * A;
rot_f4r1(3,:) = rot_f4r1(3,:) * A;
rot_f4r1(4,:) = rot_f4r1(4,:) * A;
rot_f4r1 = rot_f4r1(:,1:3);
function [rot_f4r, rot_f4r1, A] = get_model3(x, A1)
%%

fa  = 20*pi/180;

f1a = 22.5*pi/180; % 22.5*pi/180
f2a = 0*pi/180;
f3a = -45*pi/180; % -45*pi/180;
fr  = 8.15; %  8.15;
fr4 = 7.7; % 7.7

allf4a = [...
    0    0       1;    ...
    0    0       1;    ...
    0    0       1;    ...
    0 -sin(fa)  cos(fa)  ...
    ];

allf4r = [...
    -fr*sin(f1a)  fr*cos(f1a)     0;   ...
    -fr*sin(f2a)  fr*cos(f2a)     0;   ...
    -fr*sin(f3a)  fr*cos(f3a)     0;    ...
    fr4    2     0    ...
    ];

rot_f4r   = allf4r;
rot_f4r1  = rot_f4r + allf4a;

rot_f4r(:,4) = 1;
rot_f4r1(:,4) = 1;

A = A1 * hmatrix_translate(x(1:3));

rot_f4r(1,:) = rot_f4r(1,:) * A;
rot_f4r(2,:) = rot_f4r(2,:) * A;
rot_f4r(3,:) = rot_f4r(3,:) * A;
rot_f4r(4,:) = rot_f4r(4,:) * A;
rot_f4r = rot_f4r(:,1:3);

rot_f4r1(1,:) = rot_f4r1(1,:) * A;
rot_f4r1(2,:) = rot_f4r1(2,:) * A;
rot_f4r1(3,:) = rot_f4r1(3,:) * A;
rot_f4r1(4,:) = rot_f4r1(4,:) * A;
rot_f4r1 = rot_f4r1(:,1:3);

function s = cost_func(x, XYZ, A1)

[rot_f4r, rot_f4r1] = get_model6(x, A1);

s = 0;
n = size(XYZ,1);
for ii=1:n
    v1 = geom_point2line(XYZ(ii,:), rot_f4r(1,:), rot_f4r1(1,:));
    v2 = geom_point2line(XYZ(ii,:), rot_f4r(2,:), rot_f4r1(2,:));
    v3 = geom_point2line(XYZ(ii,:), rot_f4r(3,:), rot_f4r1(3,:));
    v4 = geom_point2line(XYZ(ii,:), rot_f4r(4,:), rot_f4r1(4,:));
    s = s + min([v1,v2,v3,v4]);
end
s = s / n;

function draw1D(fiducials)
figure(500); clf
k = sum(fiducials(:));
plot(squeeze(sum(sum(fiducials, 1), 2))); hold on
plot(squeeze(sum(sum(fiducials, 2), 3))+k/10); hold on
plot(squeeze(sum(sum(fiducials, 1), 3))+k/20); hold on
axis tight
grid on

return
