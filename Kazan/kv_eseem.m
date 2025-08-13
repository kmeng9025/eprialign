function [rf,spec] = kv_eseem(Sys,Exp,Opt)

out = eseem(Sys,[],Exp,Opt);
rf = out.f(end-Exp.nPoints+1: end);
spec = abs(out.fd(end-Exp.nPoints+1: end));

% ESEEM     Simulate ESEEM time-domain traces
%
%     out = eseem(nSys,eSys,Exp,Opt)
%
%     nSys  ... spin system with ESEEM nuclei only
%     eSys  ... spin system with electron spins, used for
%               orientation selection
%     Exp   ... experimental parameters
%     Opt   ... simulation options
%
%     out:
%       t   ... time axis
%       td  ... time-domain signal
%       f   ... frequency axis
%       fd  ... spectrum

function varargout = eseem(Sys,eSys,Exp,Opt)

if (nargin==0)
  %help(mfilename); return;
end

% Electronic system structure
%===================================================================
OriSelect = ~isempty(eSys);
if (OriSelect)
  if ~isfield(eSys,'S'), eSys.S = 1/2; end
  switch numel(eSys.g)
    case 1, eSys.g = eSys.g([1 1 1]);
    case 2, eSys.g = eSys.g([1 1 2]);
    case 3,
    otherwise, error('eSys.g has wrong size.');
  end
  SimpleOriSelect = ~isfield(eSys,'Nucs') & (eSys.S==1/2);
  if ~isfield(eSys,'lw'), error('Orientation selection: eSys.lw missing.'); end
end

% Experiment structure
%===================================================================
if ~isfield(Exp,'Experiment')
  error('Exp.Experiment missing! Please choose an experiment.');
end
ExpNames = {'2pESEEM','3pESEEM'};
ExperimentID = strmatch(Exp.Experiment,ExpNames);
TDExperiment = 1;

if isempty(ExperimentID), error('Experiment not recognized.'); end

switch ExperimentID
  case {1,2}
    if ~isfield(Exp,'dt')
      error('Exp.dt is missing.');
    end
end

if ~isfield(Exp,'tau'), Exp.tau = 0; end
if ~isfield(Exp,'T'), Exp.T = 0; end
if ~isfield(Exp,'t1'), Exp.t1 = 0; end
if ~isfield(Exp,'t2'), Exp.t2 = 0; end
if ~isfield(Exp,'nPoints'), Exp.nPoints = 500; end

if (OriSelect)
  if (SimpleOriSelect)
    if ~isfield(Exp,'mwFreq')
      error('Need mwFreq, since g is given.');
    end
  end
else
  if isfield(Exp,'mwFreq')
    disp('No orientation selection, but Exp.mwFreq give: ignoring it.');
  end
end

if (ExperimentID==2) && (Exp.tau==0)
  error('For a 3pESEEM, Exp.tau must be larger than 0.');
end

% Options structure
%===================================================================
Opt.Expand = 20;
if ~isfield(Opt,'nOctants')
  error(['Please specify the number of octants in Opt.nOctants.\n Possible '...
    'values are 0 (for axial symmetry), 1, 2 and 4.']);
end
Opt.Display = (nargout==0);

if ~isfield(Opt,'nKnots')
  Opt.nKnots = 90+1;
end

Opt.Method = 'b';


% Prepare nuclear Hamiltonians
%=====================================================================
Sys.I = nucspin(Sys.Nucs);
Sys.gn = nucgval(Sys.Nucs);
nNuclei = numel(Sys.I);

if size(Sys.A,1)~=nNuclei
  error('Hyperfine array has wrong size.');
end
if size(Sys.A,2)<3
  aiso = Sys.A(:,1);
  if size(Sys.A,2)==1, T = 0; else T = Sys.A(:,2); end
  Sys.A = [aiso-T,aiso-T,aiso+2*T];
end
if ~isfield(Sys,'Apa'), Sys.Apa = zeros(nNuclei,3); end

if ~isfield(Sys,'Q'), Sys.Q = zeros(nNuclei,3); end
if size(Sys.Q,2)<3
  K = Sys.Q(:,1);
  if size(Sys.Q,2)==1
    eta = 0;
  else
    eta = Sys.Q(:,2);
    if any(eta>1)|any(eta<0), error('eta must be between 0 and 1.'); end
  end
  Sys.Q = [-(1-eta).*K, -(1+eta).*K, 2*K];
end
if ~isfield(Sys,'Qpa'), Sys.Qpa = zeros(nNuclei,3); end

Hnq = 0;
Hhf1 = 0; Hhf2 = 0; Hhf3 = 0;
Hnz1 = 0; Hnz2 = 0; Hnz3 = 0;
for k = 1:nNuclei
  Ix = sop(Sys.I,k,1);
  Iy = sop(Sys.I,k,2);
  Iz = sop(Sys.I,k,3);

  % Nuclear Zeeman -------------------------------------------------
  gamm = -Sys.gn(k)*nmagn/1e3/planck/1e6;
  Hnz1 = Hnz1 + gamm*Ix;
  Hnz2 = Hnz2 + gamm*Iy;
  Hnz3 = Hnz3 + gamm*Iz;

  % Hyperfine interaction ------------------------------------------
  Ra = erot(Sys.Apa(k,:));
  A = Ra*diag(Sys.A(k,:))*Ra';
  Hhf1 = Hhf1 + A(1,1)*Ix + A(1,2)*Iy + A(1,3)*Iz;
  Hhf2 = Hhf2 + A(2,1)*Ix + A(2,2)*Iy + A(2,3)*Iz;
  Hhf3 = Hhf3 + A(3,1)*Ix + A(3,2)*Iy + A(3,3)*Iz;

  % Nuclear quadrupole ---------------------------------------------
  if (Sys.I(k)>=1)
    Rq = erot(Sys.Qpa(k,:));
    Q = Rq*diag(Sys.Q(k,:))*Rq';
    Hnq_ = ...
      Ix*(Q(1,1)*Ix + Q(1,2)*Iy + Q(1,3)*Iz) + ...
      Iy*(Q(2,1)*Ix + Q(2,2)*Iy + Q(2,3)*Iz) + ...
      Iz*(Q(3,1)*Ix + Q(3,2)*Iy + Q(3,3)*Iz);
  else
    Hnq_ = 0;
  end
  Hnq = Hnq + Hnq_;
end
nN = prod(2*Sys.I + 1);

%if (nNuclei>1), error('Not more than 1 nucleus allowed.'); end


% Electronic Hamiltonian
%=====================================================================
if (OriSelect)
  if (SimpleOriSelect)
    %mS = [-1/2, +1/2];
    Transitions = [1 2];
    nTransitions = 1;
    ManifoldsInvolved = [1 2];
  else
    [F,Gx,Gy,Gz] = sham(eSys);
    Sx = sop(eSys,1,1);
    Sy = sop(eSys,1,2);
    Sz = sop(eSys,1,3);
  end
else
  Transitions = [1 2];
  nTransitions = 1;
  ManifoldsInvolved = [1 2];
end

% Orientation set
%=====================================================================
if ~isfield(Exp,'Orientations'), Exp.Orientations = []; end
PowderSimulation = isempty(Exp.Orientations);

if (PowderSimulation)
  [Vecs,GeomWeight] = sphgrid(Opt.nOctants,Opt.nKnots,'c');
  nOrientations = numel(GeomWeight);
else
  if size(Exp.Orientations,1)==2 & size(Exp.Orientations,2)~=2
    Exp.Orientations = Exp.Orientations.';
  end
  Vecs = ang2vec(Exp.Orientations(:,1),Exp.Orientations(:,2));
  nOrientations = size(Exp.Orientations,1);
  GeomWeight = ones(1,nOrientations);
end


% Orientation selection factors
%=====================================================================
if (PowderSimulation)
  if (OriSelect)
    if (SimpleOriSelect)
      geff = diag(eSys.g)*Vecs;
      geff = sqrt(sum(geff.^2,1));
      mwResonance = geff*bmagn*Exp.Field/1e3/planck/1e6; % MHz
      gOriSelWeight = exp(-((mwResonance-1000*Exp.mwFreq)/eSys.lw).^2);
    end
  else
    gOriSelWeight = ones(1,nOrientations);
  end
else
  gOriSelWeight = ones(1,nOrientations);
end


if TDExperiment

  switch Opt.Method
    case 'b',
      % Prepare for binning method
      nPeaks = nN.^4;
      switch ExperimentID
        case 1, % 2pESEEM
          nu = zeros(1,nPeaks);
        case 2, % 3pESEEM
          nua = zeros(1,nPeaks);
          nub = zeros(1,nPeaks);
      end
      A = zeros(1,nPeaks);
      nPointsF = Opt.Expand*(Exp.nPoints-1)+1;
      fdhr = complex(zeros(1,nPointsF)); % make sure it's complex
    otherwise
      error('Accumulation method not implemented.');
  end

else
  fd = complex(zeros(1,Exp.nPoints));
end


% Orientation loop
%=====================================================================

FoldingWarning = 0;
OriSelWeight = 1;
for iOri = 1:nOrientations
  zLab = Vecs(:,iOri);

  % Compute electronic Hamiltonian and diagonalize it
  if (OriSelect)
    if (SimpleOriSelect)
      mS = [+1/2,-1/2];
      for iM = ManifoldsInvolved
        S(:,iM) = mS(iM)*zLab;
      end
      OriSelWeight = gOriSelWeight(iOri);
    else
      H = F + Exp.Field*(zLab(1)*Gx + zLab(2)*Gy + zLab(3)*Gz);
      [eV,eE] = eig(H); eE = diag(eE);
      dE = eE(:,ones(1,length(eE)));
      dE = tril(dE-dE')-Exp.mwFreq*1e3;
      w = exp(-(dE/eSys.lw).^2);
      w(w<0.005*max(w(:))) = 0;
      [v,u,OriSelWeight] = find(w);
      nTransitions = numel(v);
      Transitions = [u,v];
      ManifoldsInvolved = unique([u;v]).';
      for iM = ManifoldsInvolved
        vec = eV(:,iM);
        S(:,iM) = [vec'*Sx*vec; vec'*Sy*vec; vec'*Sz*vec];
      end
    end
  else
    mS = [+1/2,-1/2];
    for iM = ManifoldsInvolved
      S(:,iM) = mS(iM)*zLab;
    end
  end
  
    % Compute and diagonalize nuclear Hamiltonians
    Hnuc = Exp.Field*(zLab(1)*Hnz1 + zLab(2)*Hnz2 + zLab(3)*Hnz3) + Hnq;
    for iM = ManifoldsInvolved
      H = Hnuc + S(1,iM)*Hhf1 + S(2,iM)*Hhf2 + S(3,iM)*Hhf3;
      [V{iM},EE] = eig(H);
      E{iM} = diag(EE);
    end

    % Loop over all excited electronic transitions
    for iT = 1:nTransitions
      b = Transitions(iT,1); % lower manifold
      a = Transitions(iT,2); % upper manifold
      M = V{a}'*V{b}; % <a|b> overlap matrix
      Mt = M';
      Ea = E{a};
      Eb = E{b};

      % Compute peaks and amplitudes
      switch (ExperimentID)

        case 1, % 2p tau dependence (2pESEEM)

          iPeak = 1;
          for i = 1:nN
            for j = 1:nN
              Mtij = Mt(i,j);
              for k = 1:nN
                MtijMjk = Mtij*M(j,k);
                for l = 1:nN
                  nu(iPeak) = Eb(i) - Eb(k) + Ea(j) - Ea(l);
                  A(iPeak) = MtijMjk*Mt(k,l)*M(l,i);
                  iPeak = iPeak + 1;
                end
              end
            end
          end
          A = GeomWeight(iOri)*OriSelWeight(iT)*(1/2/nN)*A;

          Pdat{1} = nu;
          if (Exp.tau>0)
            Adat{1} = A.*exp(-2i*pi*Exp.tau*nu);
          else
            Adat{1} = A;
          end

        case 2, % 3p T dependence (3pESEEM)

          iPeak = 1;
          for i = 1:nN
            for j = 1:nN
              Mtij = Mt(i,j);
              for k = 1:nN
                MtijMjk = Mtij*M(j,k);
                for l = 1:nN
                  nua(iPeak) = Ea(j) - Ea(l);
                  nub(iPeak) = Eb(i) - Eb(k);
                  A(iPeak) = MtijMjk*Mt(k,l)*M(l,i);
                  iPeak = iPeak + 1;
                end
              end
            end
          end
          A = GeomWeight(iOri)*OriSelWeight(iT)*(1/8/nN)*A...
            .*exp(-2i*pi*Exp.tau*(nua+nub));
          Pdat{1} = nua;
          Pdat{2} = nub;
          if (Exp.T>0)
            Adat{1} = A.*exp(-2i*pi*nua*Exp.T);
            Adat{2} = A.*exp(-2i*pi*nub*Exp.T);
          else
            Adat{1} = A;
            Adat{2} = A;
          end

      end

      % Accumulate peaks
      %----------------------------------------------------------
      % Run over all CTPs
      if TDExperiment

        for iCTP = 1:numel(Pdat)

          if ~FoldingWarning && any(abs(Pdat{iCTP})>1/2/Exp.dt)
            FoldingWarning = 1;
          end

          switch Opt.Method
            case 'b'
              % Scale and alias peak positions P to range [0,nPoints-1)
              Pidx = 1 + fix(mod(-Pdat{iCTP}*Exp.dt,1)*nPointsF);
              turbobin1d(fdhr,Pidx,Adat{iCTP});
            otherwise
              error('Accumulation method not implemented.');
          end
        end
      else
        Pidx = 1 + fix(length(fd)*(Pdat{1}-Exp.Range(1))/(Exp.Range(2)-Exp.Range(1)));
        Pidx(Pidx<1 | Pidx>length(fd)) = [];
        turbobin1d(fd,Pidx,Adat{1});
      end

    end % electronic transition loop


end % orientation loop
%--------------------------------------------------------------

if TDExperiment
  if FoldingWarning
    disp('Warning: some peaks are outside the spectrum and folded back. Use smaller Exp.dt to avoid.');
  end

  switch Opt.Method
    case 'b'
      % Generate TD signal
      %fdhr = real(fdhr);
      td = ifft(fdhr)*numel(fdhr);
      td = real(td(1:Exp.nPoints));
    otherwise
      error('Accumulation method not implemented.')
  end


  t = (0:Exp.nPoints-1)*Exp.dt;

  Decay = 0;
  switch ExperimentID
    case 1
      if isfield(Exp,'T2')
        if (Exp.T2>0)
          td = td.*exp(-t/Exp.T2);
          Decay = 1;
        else
          disp('Warning: Exp.T2 is zero, skipping.');
        end
      end
    case 2
      if isfield(Exp,'T1')
        if (Exp.T1>0)
          td = td.*exp(-t/Exp.T1);
          Decay = 1;
        else
          disp('Warning: Exp.T1 is zero, skipping.');
        end
      end
  end

  td = td/sum(GeomWeight);
else

  fd = real(fd);

end

%===============================================================
% TD data procressing
%===============================================================

if TDExperiment
  % Baseline correction
  if (Decay)
    [kk,cc,tdfit] = exponfit(t,td,1);
    tdx = td - tdfit;
    tdx = tdx -  mean(tdx);
  else
    tdx = td -  mean(td);
  end
  %tdx(1) = tdx(1)/2;

  % Apodization
  win = apowin('han+',numel(tdx)).';
  tdx = tdx.*win;

  % Generate FD signal
  fd = fft(tdx,2*numel(tdx));
  fd = fftshift(fd);
  %================================================================

  f = fdaxis(Exp.dt,numel(fd));

  % Linear phase correction
  %fd = fd.*exp(-2i*pi*f*(Exp.tau+Exp.T));

  fdhr(1) = 0;
  %out.fdhr = fdhr;
  out.t = t;
  out.td = td;
  out.f = f;
  out.fd = fd;
else

  f = linspace(Exp.Range(1),Exp.Range(2),Exp.nPoints);

  out.t = [];
  out.td = [];
  out.f = f;
  out.fd = fd;
end

if (Opt.Display)
  clf

  if TDExperiment
    h_td = subplot(2,1,1);
    h_fd = subplot(2,1,2);

    axes(h_td);
    switch ExperimentID
      case 1
        xax = out.t + Exp.tau;
      case 2
        xax = out.t + Exp.T + Exp.tau;
      case 3
        xax = out.t + Exp.tau;
    end
    plot(xax,out.td); axis tight
    xl = xlim; xlim([0 xl(2)]);
    switch ExperimentID
      case 1,
        if Exp.tau>0
          line([1 1]*Exp.tau,ylim,'Color','k');
        end
      case 2
        line([1 1]*Exp.tau,ylim,'Color',[1 1 1]*0.7,'LineStyle',':');
        line([1 1]*(Exp.tau+Exp.T),ylim,'Color','r');
      case 3
        line([1 1]*Exp.tau,ylim,'Color',[1 1 1]*0.7,'LineStyle',':');
        line([1 1]*(Exp.tau+Exp.t2),ylim,'Color','r');
    end
    xlb = {'\tau [\mus]','\tau+T [\mus]','\tau+t_2[\mus]'};
    xlabel(xlb{ExperimentID});
    title([ExpNames{ExperimentID},', TD signal (real)']);
    set(gca,'Layer','top');

    axes(h_fd);
    idx = find(out.f==0):length(out.f);
    xf = out.f(idx);
    plot(xf,real(out.fd(idx)),'b',...
      xf, abs(out.fd(idx)),'g');
    xlim([0 max(out.f)]);
    xlabel('\nu [MHz]');
    title('Spectrum');
    legend('real','abs');
    nuI = nmagn*Sys.gn*Exp.Field/1e3/planck/1e6;
    for k=1:nNuclei
      line([1 1]*abs(nuI(k)),ylim,'Color',[1 1 1]*0.8);
      line([1 1]*2*abs(nuI(k)),ylim,'Color',[1 1 1]*0.8,'LineStyle',':');
    end
    h = get(gca,'Children');
    set(gca,'Children',h(end:-1:1));

  else

    plot(f,fd);
    xlabel('frequency [MHz]');
    title('ENDOR');
  end

elseif (nargout==1)

  varargout = {out};

end
