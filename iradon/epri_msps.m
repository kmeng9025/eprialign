function [out] = epri_msps(out, FBP)

if strcmp(safeget(FBP, 'projection_order', 'asis'), 'msps')
  nP = out.nP;
  idxcode = ['MSPS',num2str(out.nPolar),'index.mat'];
  if exist(idxcode, 'file')
    s1 = load(idxcode);
    mspsidx = s1.idx;
  else
    mspsidx = 1:nP;
    warning('MSPS index was not loaded.');
  end
  out.mspsidx = mspsidx;
  out.G = out.G(mspsidx, :);
  out.UnitSweep = out.UnitSweep(mspsidx);
  
  % out.nTrace did not change
end