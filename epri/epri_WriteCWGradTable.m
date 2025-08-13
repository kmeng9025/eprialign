function epri_WriteCWGradTable(fname, pars, ext_pars)

dummy_projections = safeget(pars, 'dummy_projections', 0);
fid = fopen(fname, 'w');

if fid ~= -1
  npoints = 256;
  full_spectrum_acquisition = 1;
  angle_coverage_method = 2;
  coord_pole = 'Z';
  SW_definition = 0;

  FBP = pars.data.FBP;

  idx = 1:pars.nP;
  FullIdx = 1:pars.nTrace;
  FullIdx(pars.service_idx ~= -1) = idx;
  FullIdx(pars.service_idx == -1) = -1;
  
  fprintf(fid, "%d %% Number of spatial polar angles\n", pars.nPolar);
  fprintf(fid, "%d %% Number of spatial azimuthal angles\n", pars.nAz);
  fprintf(fid, "%d %% Number of acquired spectral angles\n", pars.nSpec);
  fprintf(fid, "%d %% Number of points\n", npoints);
  fprintf(fid, "%d %% Full Spectral acquisition (0/1)\n", full_spectrum_acquisition);
  fprintf(fid, "%d %% Image type\n", FBP.imtype);
  fprintf(fid, "%d %% Number of spectral angles over 180\n", pars.nSpec);
  fprintf(fid, "%d %% Uniform Solid angles (0/1)\n", angle_coverage_method);
  fprintf(fid, "%d %% SW definition\n", SW_definition);
  fprintf(fid, "%5.3g %% Spectral FOV (delta H in G)\n", pars.deltaH);
  fprintf(fid, "%5.3g %% Spatial FOV (delta L in cm)\n", pars.deltaL);
  fprintf(fid, "%s %% Coord. Pole\n", coord_pole);
  % PolAzSpec, nScan, gradx, grady, gradz, alpha, sweep, swFraction, nPts

  fprintf(fid, "%% PolAzSpec, nScan, gradx, grady, gradz, alpha, sweep, swFraction, nPts\n");
  for ii=1:pars.nTrace
    sequence_index = FullIdx(ii);
    if sequence_index ~= -1
      alpha = ext_pars.alpha(sequence_index);
      projection =  ext_pars.i(sequence_index)*10000 + ext_pars.j(sequence_index)*100 + ext_pars.k(sequence_index);
    else
      alpha = 0;
      projection = 0;
    end
    fprintf(fid, "    %06d, %3d, %6.3f, %6.3f, %6.3f, %6.4f, %6.4g, %4.2g, %d\n", ...
      projection, ...
      fix(pars.N/(min(pars.Sweep)/pars.Sweep(ii))), ...
      pars.G(ii,1), pars.G(ii,2), pars.G(ii,3),...
      alpha,...
      pars.Sweep(ii),...
      1,...
      npoints);
  end
  for ii=1:dummy_projections
    dummy = 1;
    sequence_index = FullIdx(dummy);
    if sequence_index ~= -1
      alpha = ext_pars.alpha(sequence_index);
      projection =  ext_pars.i(sequence_index)*10000 + ext_pars.j(sequence_index)*100 + ext_pars.k(sequence_index);
    else
      alpha = 0;
      projection = 0;
    end
    fprintf(fid, "    %06d, %3d, %6.3f, %6.3f, %6.3f, %6.4f, %6.4g, %4.2g, %d\n", ...
      projection, ...
      fix(pars.N/(min(pars.Sweep)/pars.Sweep(dummy))), ...
      pars.G(dummy,1), pars.G(dummy,2), pars.G(dummy,3),...
      alpha,...
      pars.Sweep(dummy),...
      1,...
      npoints);
  end
  fclose(fid);
end
