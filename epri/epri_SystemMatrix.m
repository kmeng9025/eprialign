function M = epri_SystemMatrix(k, prj, l1, l2, l3, b1)

nK   = length(k);   % number of projections
nPts = length(prj); % number of points on projection

% one D
if isempty(l2) && isempty(l3) && isempty(b1)
    nR = length(l1);
    M = zeros(nK, nPts, nR);
    for kk=1:nK
        pos = k(kk)*l1;
        for ii=1:nR % per point
            [~,idx] = min(abs(prj-pos(ii)));
            M(kk, idx, ii) = 1;
        end
    end
% 2D
elseif isempty(l3) && isempty(b1)
    nR = length(l1)*length(l2);
    [L1,L2] = meshgrid(l1,l2);
    M = zeros(nK, nPts, nR);
    for kk=1:nK
        pos = k(kk,1)*L1+k(kk,2)*L2;
        pos = pos(:);
        for ii=1:nR % per point
            [~,idx] = min(abs(prj-pos(ii)));
            M(kk, idx, ii) = 1;
        end
    end
% 3D
elseif isempty(b1)
    nR = length(l1)*length(l2)*length(l3);
    [L1,L2,L3] = meshgrid(l1,l2,l3);
    M = zeros(nK, nPts, nR);
    for kk=1:nK
        pos = k(kk,1)*L1+k(kk,2)*L2+k(kk,3)*L3;
        pos = pos(:);
        for ii=1:nR % per point
            [~,idx] = min(abs(prj-pos(ii)));
            M(kk, idx, ii) = 1;
        end
    end
% 4D    
else
    nR = length(l1)*length(l2)*length(l3);
    nB = length(b1);
    [L1,L2,L3] = meshgrid(l1,l2,l3);
    M = zeros(nK, nPts, nR, nB);
    for kk=1:nK
        pos = k(kk,1)*L1+k(kk,2)*L2+k(kk,3)*L3;
        pos = pos(:);
        for ii=1:nR % per point
            for jj=1:nB
                [~,idx] = min(abs(prj-pos(ii)-b1(jj)));
                M(kk, idx, ii, jj) = 1;
            end
        end
    end
    M = reshape(M, [nK, nPts, nR * nB]);
end