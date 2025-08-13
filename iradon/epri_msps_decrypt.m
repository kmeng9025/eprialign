function [P1, out] = epri_msps_decrypt(P, out, FBP)

if strcmp(safeget(FBP, 'projection_order', 'off'), 'msps')
  idxcode = ['MSPS',num2str(FBP.nPolar),'index.mat'];
  if exist(idxcode, 'file')
    s1 = load(idxcode);
    out.mspsidx = s1.idx;
  else
    out.mspsidx = 1:FBP.nP;
    warning('MSPS index was not loaded.');
  end
  if isfield(out, 'G')
    out.G(out.mspsidx, :) = out.G;
  end
  if isfield(out, 'Gexp')
    out.Gexp(out.mspsidx, :) = out.Gexp;
  end
  if isfield(out, 'UnitSweep')
    out.UnitSweep(out.mspsidx) = out.UnitSweep;
  end
  P1 = zeros(size(P));
  P1(:,:,out.mspsidx) = P;
else
  P1 = P;
end