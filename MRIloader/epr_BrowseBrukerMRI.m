function [res]=epr_BrowseBrukerMRI(the_path)

if isempty(the_path)
  the_path = uigetdir(the_path);
end

res.given_path = the_path;
res.root_path = find_root(the_path);

if ~isempty(res.root_path)
  % Look for data structure
  rs = dir(res.root_path);
  rs = rs(vertcat(rs.isdir));
  for jj=1:length(rs)
    mri_folder = str2double(rs(jj).name);
    if  mri_folder > 0
      res.data{mri_folder}.folder = [res.root_path , filesep, rs(jj).name];
      a = epr_ReadBrukerAcqp([res.data{mri_folder}.folder, filesep, 'acqp']);
      res.data{mri_folder}.protocol = a.ACQ_protocol_name;
      
      switch upper(a.PULPROG)
        case 'RARE.PPG',  res.data{mri_folder}.method = 'RARE imaging experiment';
        case 'FLASH.PPG',  res.data{mri_folder}.method = 'FLASH - a gradient echo imaging method';
        otherwise,  res.data{mri_folder}.method ='Unknown';
      end
      j1 = a.ACQ_grad_matrix(1:3);
      
    switch 'yes' % lower(saddleCoil)
        case 'yes'
            % acqpInfo.SaddleCoil='Yes';
            % saddle coil, axial is sagittal & vice versa
            if j1(1)==1
                res.data{mri_folder}.imageType='Axial';
            elseif j1(2)==1
                res.data{mri_folder}.imageType='Coronal';
            elseif j1(3)==1
                res.data{mri_folder}.imageType='Sagittal';
            end
        case 'no'
            if j1(1)==1
                res.data{mri_folder}.imageType='Sagittal';
            elseif j1(2)==1
                res.data{mri_folder}.imageType='Coronal';
            elseif j1(3)==1
                res.data{mri_folder}.imageType='Axial';
            end
    end % case
      
    end
  end
  
  str = {};
  for ii=1:length(res.data)
    str{ii} = res.data{ii}.protocol;
  end
  res.Selection = listdlg('PromptString', 'Select and image', 'ListString', str, 'SelectionMode', 'single');
end

function directoryname = find_root(directoryname)

if ispc
  parts = strsplit(directoryname, '\\');
  the_root = parts{1};
  parts = parts(2:end);
else
  parts = strsplit(directoryname, '/');
end

for ii=length(parts):-1:0
  directoryname = the_root;
  if ii > 0
    for jj=1:ii
      directoryname =[directoryname , filesep, parts{jj}]; %#ok<AGROW>
    end
  end
  
  %   disp(directoryname)
  %   look for 1/2/3/4... directory structure   
  rs = dir(directoryname);
  rs = rs(vertcat(rs.isdir));
  
  for jj=1:length(rs)
    mri_folder = str2double(rs(jj).name);
    if  mri_folder > 0
      % Additional check
      if exist([directoryname, filesep, 'subject'], 'file')
        return;
      else
        directoryname = [];
        return;
      end
    end
  end
end

