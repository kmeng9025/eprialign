function imrt_openscad(the_type, Target, Scale_factor, UVShift, Plug_size, scad_file)

%Path to an openscad so we don't have version control issues.
Openscad_path = 'z:\CenterHardware\3Dprint\OpenSCAD';
% Openscad_path = 'C:\Program Files\OpenSCAD';

Openscad_command_str = 'openscad';

switch the_type
  case 'beam'
    %imagesc(Target_dil)
    Bound_regions = bwboundaries(Target,'noholes');
    
    Poly_vector = {};
    for k = 1:length(Bound_regions)
      boundary = Bound_regions{k};
      %plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
      
      Poly_vector{k}(:,1) =  boundary(:,1) - size(Target, 1)/2;
      Poly_vector{k}(:,2) =  boundary(:,2) - size(Target, 2)/2;
    end
    
    disp('Writing SCAD files and rendering plugs for Boost')
    
    switch Plug_size
      case '16mm', fn = @epr_Contour2OpenSCAD_Plugs;
      case '18mm', fn = @epr_Contour2OpenSCAD_Plugs_18_mm;
      case '20mm', fn = @epr_Contour2OpenSCAD_Plugs_20_mm;
      case '22mm', fn = @epr_Contour2OpenSCAD_Plugs_22_mm;
    end
    
    scad_filename = [scad_file,'.scad'];
    stl_filename = [scad_file,'.stl'];
    fn('Plug_cut', ...
      scad_filename, Poly_vector, Scale_factor, UVShift, 0);
    fprintf('SCAD file has been written to %s.\n',scad_filename);
    
    %Render plugs
    cd(Openscad_path) %Cd to the sharedrive openscad location.
    dos(sprintf('%s -o %s %s',Openscad_command_str,stl_filename,scad_filename));

    case 'inv-beam'
    %imagesc(Target_dil)
    Bound_regions = bwboundaries(Target,'noholes');
    
    Poly_vector = {};
    for k = 1:length(Bound_regions)
      boundary = Bound_regions{k};
      %plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
      
      Poly_vector{k}(:,1) =  boundary(:,1) - size(Target, 1)/2;
      Poly_vector{k}(:,2) =  boundary(:,2) - size(Target, 2)/2;
    end
    
    disp('Writing SCAD files and rendering plugs for Boost')
    
    switch Plug_size
      case '16mm', fn = @epr_Contour2OpenSCAD_Plugs;
      case '18mm', fn = @epr_Contour2OpenSCAD_Plugs_18_mm;
      case '20mm', fn = @epr_Contour2OpenSCAD_Plugs_20_mm;
      case '22mm', fn = @epr_Contour2OpenSCAD_Plugs_22_mm;
    end
    
    scad_filename = [scad_file,'.scad'];
    stl_filename = [scad_file,'.stl'];
    fn('Plug_cut', ...
      scad_filename, Poly_vector, Scale_factor, UVShift, 1.5);
    fprintf('SCAD file has been written to %s.\n',scad_filename);
    
    %Render plugs
    cd(Openscad_path) %Cd to the sharedrive openscad location.
    dos(sprintf('%s -o %s %s',Openscad_command_str,stl_filename,scad_filename));
    
  case 'shell'
    Bound_regions = bwboundaries(Target,'holes');
    %[Bound_regions_support,L] = bwboundaries(Support,'holes');
    
    %imshow(BW)
    %hold on
    Poly_vector = cell(length(Bound_regions), 1); %Convert to centered rather than vectored input
    for k = 1:length(Bound_regions)
      boundary = Bound_regions{k};
      %plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
      
      Poly_vector{k}(:,1) =  boundary(:,1) - size(Target, 1)/2;
      Poly_vector{k}(:,2) =  boundary(:,2) - size(Target, 2)/2;
    end
    
    %investigate the Poly_vector of the mask. Identify any outlines
    %which are internal. These need to be classified differenly so they are added to the plug instead of subtracted.
    %MM 8/30/2016 commented this code out. This way is flawed in that
    %two structural elements can cause each other to be assumed to be
    %outline elements. Rewritten below:
    %         Poly_vectors_struct = {};
    %         for k = 1:length(Bound_regions)
    %         Outline = zeros(size(Target));
    %         idx = sub2ind(size(Target),Bound_regions{k}(:,1),Bound_regions{k}(:,2));
    %         Outline(idx) = 1;
    %         Outline_filled = imfill(Outline,'holes');
    %         if numel(find(Target(find(Outline_filled-Outline))))>1;
    %         disp('outside border')
    %         else
    %         disp('inside border adding to structure')
    %         Poly_vectors_struct{end+1} = Bound_regions{k};
    %         end
    %         end
    
    %investigate the Poly_vector of the mask. Identify any outlines
    %use a priori statment: There is only one outline, and it is the
    %largest area when filled.
    
    Outline_area_max = 0;
    for k = 1:length(Bound_regions)
      Outline = zeros(size(Target));
      idx = sub2ind(size(Target),Bound_regions{k}(:,1),Bound_regions{k}(:,2));
      Outline(idx) = 1;
      Outline_filled = imfill(Outline,'holes');
      if numel(find(Outline_filled)) > Outline_area_max
        Outline_idx = k; Outline_area_max  = numel(find(Outline_filled));
      end
    end
    %         Poly_vectors_struct = Bound_regions{k};
    %         Poly_vectors_struct{Outline_idx} = [];
    
    Poly_vectors_struct = Bound_regions;
    Poly_vectors_struct{Outline_idx} = [];
    
    %figure; imagesc(Target); hold on; for vect = 1:length(Poly_vectors_struct); plot(Poly_vectors_struct{vect}(:,2),Poly_vectors_struct{vect}(:,1), 'm');end
    
    Poly_vectors_structure ={}; %Convert to centered rather than vectored input
    if ~isempty(Poly_vectors_struct)
      for k = 1:length(Poly_vectors_struct)
        if length(Poly_vectors_struct{k}) > 1
          boundary = Poly_vectors_struct{k};
          %plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
          Poly_vectors_structure{k}(:,1) =  boundary(:,1) - size(Target, 1)/2;
          Poly_vectors_structure{k}(:,2) =  boundary(:,2) - size(Target, 2)/2;
        else
          Poly_vectors_structure{k} = [0 0];
        end
      end
    else
      Poly_vectors_structure = {[0 0]};
    end
    
    switch Plug_size
      case '16mm', fn = @epr_Contour2OpenSCAD_Plugs_1_full_layer_plus_structure;
      case '18mm', fn = @epr_Contour2OpenSCAD_Plugs_1_full_layer_plus_structure_18_mm;
      case '20mm', fn = @epr_Contour2OpenSCAD_Plugs_1_full_layer_plus_structure_20_mm;
      case '22mm', fn = @epr_Contour2OpenSCAD_Plugs_1_full_layer_plus_structure_22_mm;
    end
    
    scad_filename = [scad_file,'.scad'];
    stl_filename = [scad_file,'.stl'];
    fn('Plug_cut','Plug_structures', ...
      scad_filename, Poly_vector,Poly_vectors_structure, ...
      Scale_factor, UVShift);
    fprintf('SCAD file has been written to %s.\n',scad_filename);
    
    
    %Render plugs
    cd(Openscad_path) %Cd to the sharedrive openscad location.
    dos(sprintf('%s -o %s %s',Openscad_command_str,stl_filename,scad_filename));
end
