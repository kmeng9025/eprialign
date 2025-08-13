% Jontahan Bryant, 2012

function [mat,Pars,Rec] = Reconstruct(p, Pars, Rec)

  switch Pars.nDim
    case 3
      [Pars.nBins, Pars.nProj] = size(p);
      [Rec] = geo_getangles3D(Pars,Rec);
      [p,Pars,Rec] = interp_tri3D(p,Pars,Rec);
      [p,Pars,Rec] = geo_Weights3D(p,Pars,Rec);
      [p] = recon_FilterProjections(p,Pars,Rec);
      switch Rec.Process
        case 'GPU'
        [mat] = recon_bp3d_gpu(p,Pars,Rec);
        case 'CPU'
        [mat] = recon_bp3d(p,Pars,Rec);
        case 'CPU Single'
      end
    case 4
      switch Rec.Method
        case 'FBP'
          p = recon_FilterProjections(p,Pars,Rec);
          mat = recon_bp4d_gpu(p,Pars,Rec);
        case 'BPF'
          p_hil = recon_FilterChords(p,Pars,Rec);
          mat_hil = recon_bp4d_single(p_hil,Pars,Rec);
          mat = recon_invhilbertNd(mat_hil);
      end
  end
