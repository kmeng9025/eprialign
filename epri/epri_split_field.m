function  pars = epri_split_field(pars, FBP)

Bvals = safeget(FBP, 'split_field', []);
if isempty(Bvals), return; end

nmb_idx = length(Bvals);
pars.gidx = split_ABC_AA_BB_CC(pars.gidx, nmb_idx);
pars.GradX = split_ABC_AA_BB_CC(pars.GradX, nmb_idx);
pars.GradY = split_ABC_AA_BB_CC(pars.GradY, nmb_idx);
pars.GradZ = split_ABC_AA_BB_CC(pars.GradZ, nmb_idx);
pars.UnitSweep = split_ABC_AA_BB_CC(pars.UnitSweep, nmb_idx);

pars.Offset = repmat(Bvals(:), 1, pars.nTrace);
pars.Offset = pars.Offset(:);

pars.nTrace = length(pars.gidx);

% split each setting in two following each other
function P = split_ABC_AA_BB_CC(P, n_splits)
P = repmat(P(:), 1, n_splits)';
P = P(:);