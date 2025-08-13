% pars = GetSPIGradientTable(SPI)

function pars = td_GetSPIGradientTable(SPI)

Method = safeget(SPI, 'Method','8Q_flip_optimization');
Grad   = safeget(SPI, 'MaxGradient', 1);

switch Method
  case {'8Q_no_optimization','8Q_flip_optimization','4Q_flip_optimization'}
    Steps  = safeget(SPI, 'nSteps', 15);
    Steps  = safeget(SPI, 'nSteps', ones(1, 3)*Steps(1));
    case 'Full_Rectangular'
       Steps  = safeget(SPI, 'nSteps', 15);
end

switch Method
    case '8Q_no_optimization'
%         Grad1D = linspace(-Grad, Grad, Steps)';
        Grad1D = linspace(-Grad, Grad, Steps(1))';
        Grad3Dx = Grad1D(:, ones(Steps(1),1), ones(Steps(1),1));
        Grad3Dy = permute(Grad3Dx, [2,1,3]);
        Grad3Dz = permute(Grad3Dx, [3,2,1]);

        GradR = sqrt(Grad3Dx(:).*Grad3Dx(:) + Grad3Dy(:).*Grad3Dy(:) + Grad3Dz(:).*Grad3Dz(:));

        gidx    =   find(GradR < Grad);
        pars.nP = length(gidx);
        pars.gidx = gidx;

        GradX = Grad3Dx(gidx);
        GradY = Grad3Dy(gidx);
        GradZ = Grad3Dz(gidx);

        idx = (1:Steps)';
        idxx = idx(:, ones(Steps(1),1), ones(Steps(1),1));
        idxy = permute(idxx, [2,1,3]);
        idxz = permute(idxx, [3,2,1]);

        pars.pidx.i = idxx(gidx);
        pars.pidx.j = idxy(gidx);
        pars.pidx.k = idxz(gidx);
    case '8Q_flip_optimization'
        Grad1D = linspace(-Grad, Grad, Steps(1))';
        Grad3Dx = Grad1D(:, ones(Steps(2),1), ones(Steps(3),1));
        Grad1D = linspace(-Grad, Grad, Steps(2));
        Grad3Dy = Grad1D(ones(Steps(1),1), :, ones(Steps(3),1));
        Grad1D = reshape(linspace(-Grad, Grad, Steps(3)), [1,1,Steps(3)]);
        Grad3Dz = Grad1D(ones(Steps(1),1), ones(Steps(2),1), :);

        Grad3Dx(:,1:2:end,:) = Grad3Dx(end:-1:1,1:2:end,:);
        Grad3Dy(:,1:end,1:2:end) = Grad3Dy(:,end:-1:1,1:2:end);

        GradR = sqrt(Grad3Dx(:).*Grad3Dx(:) + Grad3Dy(:).*Grad3Dy(:) + Grad3Dz(:).*Grad3Dz(:));

        gidx    =   find(GradR <= Grad);
        pars.nP = length(gidx);

        GradX = Grad3Dx(gidx);
        GradY = Grad3Dy(gidx);
        GradZ = Grad3Dz(gidx);

        idx = (1:Steps(1))';
        idxx = idx(:, ones(Steps(2),1), ones(Steps(3),1));
        idx = 1:Steps(2);
        idxy = idx(ones(Steps(1),1), :, ones(Steps(3),1));
        idx = reshape(1:Steps(3), [1,1,Steps(3)]);
        idxz = idx(ones(Steps(1),1), ones(Steps(2),1), :);
        
        idxx(:,1:2:end,:) = idxx(end:-1:1,1:2:end,:);
        idxy(:,1:end,1:2:end) = idxy(:,end:-1:1,1:2:end);

        pars.pidx.i = idxx(gidx);
        pars.pidx.j = idxy(gidx);
        pars.pidx.k = idxz(gidx);
        
        gidx = pars.pidx.i + (pars.pidx.j-1)*Steps(1) + (pars.pidx.k-1)*Steps(1)*Steps(2);
        pars.gidx = gidx;
    case '4Q_flip_optimization',
        Grad1D = linspace(-Grad, Grad, Steps(1));
        Gstep = Grad / (Steps(1) - 1);

        Grad3Dx = Grad1D(ones(Steps(1),1), :, ones(Steps(1),1));
        Grad3Dy = permute(Grad3Dx, [2,3,1]);
        Grad3Dz = permute(Grad3Dx, [3,1,2]);

        Grad3Dy(:,2:2:end,:) = Grad3Dy(end:-1:1,2:2:end,:);
        Grad3Dx(:,1:end,2:2:end) = Grad3Dx(:,end:-1:1,2:2:end);
        Grad3Dy(1:end,:,2:2:end) = Grad3Dy(end:-1:1,:,2:2:end);

        GradR = sqrt(Grad3Dx(:).*Grad3Dx(:) + Grad3Dy(:).*Grad3Dy(:) + Grad3Dz(:).*Grad3Dz(:));

        gidx    =   find(GradR <= Grad & Grad3Dz(:) >= 0);
        pars.nP = length(gidx);

        GradX = Grad3Dx(gidx);
        GradY = Grad3Dy(gidx);
        GradZ = Grad3Dz(gidx);

        idx = 1:Steps;
        idxx = idx(ones(Steps,1), :, ones(Steps,1));
        idxy = permute(idxx, [2,3,1]);
        idxz = permute(idxx, [3,1,2]);
        idxy(:,1:2:end,:) = idxy(end:-1:1,1:2:end,:);
        idxx(:,1:end,2:2:end) = idxx(:,end:-1:1,2:2:end);
        idxy(1:end,:,2:2:end) = idxy(end:-1:1,:,2:2:end);

        pars.pidx.i = idxx(gidx);
        pars.pidx.j = idxy(gidx);
        pars.pidx.k = idxz(gidx);

        gidx = (pars.pidx.i-1)*Steps(1) + pars.pidx.j + (pars.pidx.k-1)*Steps(1)*Steps(1);
        pars.gidx = gidx;
        
        case 'Full_Rectangular',
end

pars.deltaH = 0.512;
pars.fullSW = 0.512;
pars.UnitSweep = ones(length(GradX), 1);

pars.Dim = Steps;
pars.nTrace = pars.nP;

pars.G = [GradX,GradY,GradZ];
pars.service_idx = ones(pars.nP,1);

pars = epri_baseline(pars, SPI);
pars.data.Modality = 'PULSESPI';
pars.data.SPI = SPI;

