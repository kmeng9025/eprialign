function [shf, varargout] = cw_shf_model(varargin)
% model_list = cw_shf_model        - get the model's list
% model = cw_shf_model(model_name) - retrieve the model
% model list is a cell array of strings
% model.Name - model name
% model.n_hf - number of resolved hyperfine lines
% model.Nucs - cell array of nuclei structures
% model.Nucs fields: Name, HF, Spin, Abundance, EqNucs

% to add new model: i) add name to the list of model ii) add new case with the name and add nucleii

if nargin == 0
  shf = {'FIND',...
    'FINH',...
    'OX031H',...
    'OX031D',...
    'OX063H',...
    'OX063D24',...
    'N14',...
    'N15',...
    'Grad Calibration',...
    'lorentzian'
    };
else
  shf.Nucs={};
  shf.Name = varargin{1};
  switch upper(varargin{1})
    case 'FIND'	% Finland deuterated label
      shf.n_hf = 1;
      shf.Nucs{end+1} = Nuc2Fld('C-6-finD', 0.179, 1.1, 1/2, 6 );% S-C-S ring carbons in Finland
      shf.Nucs{end+1} = Nuc2Fld('C-12-finD', 0.0236, 1.1, 1/2, 12 );% C13 of methyl deuteron CD3
      shf.Nucs{end+1} = Nuc2Fld( 'Dmet36',.00215, 100., 1, 36);% D methyl deuterons in Finland
    case 'FINH'	% Finland protonated label
      shf.n_hf = 1;
      shf.Nucs{end+1} = Nuc2Fld('C-6-finH', 0.179, 1.1, 1/2, 6 );% S-C-S ring carbons in Finland H
      shf.Nucs{end+1} = Nuc2Fld('C-12-H', 0.0114, 1.1, 1/2, 12 );% C 13 of proton CH3 carbons
      shf.Nucs{end+1} = Nuc2Fld( 'Hmet36', .0074, 100., 1/2, 36);% methyl protons in Finland
    case 'OX031H'	% OX31 protonated label
      shf.n_hf = 1;
      shf.Nucs{end+1} = Nuc2Fld('C-6-31', 0.209, 1.1, 1/2, 6 );% S-C-S ring carbons in OX63, Ox31
      shf.Nucs{end+1} = Nuc2Fld('H1-24-31',0.0165, 100., 1/2, 24);% OX31 preO-CH2 protons  in side chain
      shf.Nucs{end+1} = Nuc2Fld('H2-24-O-31',0.004, 100., 1/2, 24);% OX31 post O-CH2 protons  in side chain
      shf.Nucs{end+1} = Nuc2Fld('C-12-notD', 0.0114, 1.1, 1/2, 12 );% C for H-1-24
    case 'OX031D'	% OX31 deuterated label
      shf.n_hf = 1;
      shf.Nucs{end+1} = Nuc2Fld('C-6-31', 0.209, 1.1, 1/2, 6 );% S-C-S ring carbons in OX63, Ox31
      shf.Nucs{end+1} = Nuc2Fld('D1-24-31',0.003, 100., 1, 24);% OX31 preO-CH2 deuterons  in side chain
      shf.Nucs{end+1} = Nuc2Fld('H2-24-O-31',0.004, 100., 1/2, 24);% OX31 post O-CH2 protons  in side chain
      shf.Nucs{end+1} = Nuc2Fld('C-12-D', 0.0114, 1.1, 1/2, 12 );% C for D-1-24
    case {'OX063H','OX063'}	% OX063 protonated
      shf.n_hf = 1;
      shf.Nucs{end+1} = Nuc2Fld('C-6-63', 0.209, 1.1, 1/2, 6 );% S-C-S ring carbons in OX63, Ox31
      shf.Nucs{end+1} = Nuc2Fld('H1-24-63',0.024, 100., 1/2, 24);% Ofirst-CH2 protons  in side chain
      shf.Nucs{end+1} = Nuc2Fld('H2-24-63',0.0175, 100., 1/2, 24);% second-CH2 protons  in side chain
      shf.Nucs{end+1} = Nuc2Fld('C-12-notD', 0.0114, 1.1, 1/2, 12 );% C for H-1-24
    case {'OX063D24','OX071'}	% OX063 partially deuterated / OX071
      shf.n_hf = 1;
      shf.Nucs{end+1} = Nuc2Fld('C-6-63', 0.2011, 1.1, 1/2, 6 );% S-C-S ring carbons in OX63, Ox31
      shf.Nucs{end+1} = Nuc2Fld('D1-24-63',0.00369, 100., 1/2, 24);% Ofirst-CH2 protons  in side chain
      shf.Nucs{end+1} = Nuc2Fld('H2-24-63',0.013, 100., 1/2, 24);% second-CH2 protons  in side chain
      shf.Nucs{end+1} = Nuc2Fld('C-12-notD', 0.0114, 1.1, 1/2, 12 );% C for H-1-24
    case'N14'
      shf.n_hf = 3;
      %shf.Nucs{end+1} = Nuc2Fld('N14', '14', '100', '1 ', 1 );%  nitrogen
      shf.Nucs{end+1} = Nuc2Fld( 'Hmet', .19, 100., 1/2, 12);% methyl protons
      %shf= str2mat(shf,'Hax','.5', 100., 1/2, 1);% axial proton
      shf.Nucs{end+1} = Nuc2Fld('Hax', .2, 100., 1/2, 2);% axial protons
      shf.Nucs{end+1} = Nuc2Fld('C-13', 7, 1.1, 1/2, 4 );% C 13 ring
      %shf= str2mat(shf,'H1rad','13', 100., 1/2, 1);%  proton
    case 'N15'
      shf.n_hf = 2;
      %shf = str2mat('N15', '22', '100', '1/2 ', 1 );%  nitrogen
      shf.Nucs{end+1} = Nuc2Fld('Hmet','.19', 100., 1/2, 12);% methyl protons
      shf.Nucs{end+1} = Nuc2Fld('Hax', .5, 100., 1/2, 1);% axial proton
      shf.Nucs{end+1} = Nuc2Fld('C-13', 7, 1.1, 1/2, 4 );% C 13 ring
      shf.Nucs{end+1} = Nuc2Fld('H1rad', 13, 100., 1/2, 1);%  proton
    case 'GRAD CALIBRATION'
      shf.n_hf = 1;
      shf.Nucs{end+1} = Nuc2Fld('gradcal', 0, 100, 0, 1 );%  nitrogen
    case 'OTHER'
      shf.n_hf = 1;
      shf.Nucs{end+1} = Nuc2Fld('N14', 13, 100, 1 , 1);
      % shf = str2mat(shf, 'Hmet','.19', 100., 1/2, 12);% methyl protons
      %	shf= str2mat(shf,'Hax','.5', 100., 1/2, 1);% axial proton
      shf.Nucs{end+1} = Nuc2Fld('H1', 12, 100., 1/2, 1);%  proton
      shf.Nucs{end+1} = Nuc2Fld('H2', 1.2, 100., 1/2, 1);%  proton
      shf.Nucs{end+1} = Nuc2Fld('C-13', 7, 1.1, 1/2, 4 );% C 13 ring
      case 'LORENTZIAN'
      shf.n_hf = 1;
      shf.Nucs{end+1} = Nuc2Fld('H', 0.0, 100, 1/2 , 1);          
  end
end

if nargout > 1
  SpSys = [];
  if isfield(shf, 'Nucs')
    SpSys = zeros(length(shf.Nucs), 4);
    for ii=1:length(shf.Nucs)
      SpSys(ii,1) = shf.Nucs{ii}.HF;
      SpSys(ii,2) = shf.Nucs{ii}.Abundance;
      SpSys(ii,3) = shf.Nucs{ii}.Spin;
      SpSys(ii,4) = shf.Nucs{ii}.EqNucs;
    end
  end
  varargout{1} = SpSys;
end

% --------------------------------------------------------------------
function Nucs = Nuc2Fld(Name, hf, Abundance, spin, equiv_nucs)
Nucs.Name = Name;
Nucs.HF = hf;
Nucs.Spin = spin;
Nucs.EqNucs = equiv_nucs;
Nucs.Abundance = Abundance;
