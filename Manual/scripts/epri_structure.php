<?
	$FOLDER = array();
	$FOLDER['root'] = '../';
	$FOLDER['content']='content/';
	$FOLDER['content2']='content/';
	
	$PAGE = array();

	//---- MAIN LOGO SECTION -----------------------------------------------------
	$PAGE['index'] = array('title'=>'About', 'parent'=>'root');
	$PAGE['general_contribution'] = array('title'=>'Contributions', 'parent'=>'root');
	$PAGE['general_installation'] = array('title'=>'Installation', 'parent'=>'root');
	
	$PAGE['sample_data'] = array('title'=>'Sample data', 'parent'=>'root');
	$PAGE['sample_dataset1'] = array('title'=>'Dataset 1', 'parent'=>'sample_data');
	$PAGE['sample_dataset2'] = array('title'=>'Dataset 2', 'parent'=>'sample_data');
	$PAGE['sample_dataset3'] = array('title'=>'Dataset 3', 'parent'=>'sample_data');
	$PAGE['sample_dataset4'] = array('title'=>'Dataset 4', 'parent'=>'sample_data');

	//---- RADON TOOLBOX -----------------------------------------------------
	$PAGE['radon_toolbox'] = array('title'=>'Radon toolbox', 'parent'=>'root');
	$PAGE['radon_c2d_sphere'] = array('title'=>'radon_c2d_sphere', 'parent'=>'radon_toolbox');
	$PAGE['radon_c2d_cube'] = array('title'=>'radon_c2d_cube', 'parent'=>'radon_toolbox');
	$PAGE['radon_d2d'] = array('title'=>'radon_d2d', 'parent'=>'radon_toolbox');
	$PAGE['radon_phantom'] = array('title'=>'radon_phantom', 'parent'=>'radon_toolbox');
	$PAGE['radon_angle2xyz'] = array('title'=>'radon_angle2xyz', 'parent'=>'radon_toolbox');

	//---- IRADON TOOLBOX -----------------------------------------------------
	$PAGE['iradon_toolbox'] = array('title'=>'Iradon toolbox', 'parent'=>'root');
	$PAGE['iradon_d2d_mstage'] = array('title'=>'iradon_d2d_mstage', 'parent'=>'iradon_toolbox');
	$PAGE['iradon_qiao_fbp3d'] = array('title'=>'iradon_qiao_fbp3d', 'parent'=>'iradon_toolbox');
	$PAGE['iradon_getcoordpole'] = array('title'=>'iradon_GetCoordPole', 'parent'=>'iradon_toolbox');
	$PAGE['iradon_getfbpimagetype'] = array('title'=>'iradon_GetFBPImageType', 'parent'=>'iradon_toolbox');
	$PAGE['iradon_fbpgradtable'] = array('title'=>'iradon_FBPGradTable', 'parent'=>'iradon_toolbox');
	$PAGE['iradon_interptouniformangle'] = array('title'=>'iradon_InterpToUniformAngle', 'parent'=>'iradon_toolbox');
	$PAGE['iradon_vor_area_3d'] = array('title'=>'iradon_vor_area_3d', 'parent'=>'iradon_toolbox');
	
	//---- COMMON TOOLBOX: LINEAR ALGEBRA---------------------------------------
	$PAGE['common_toolbox'] = array('title'=>'Common toolbox: Algebra', 'parent'=>'root');
	$PAGE['connected_components'] = array('title'=>'connected_components', 'parent'=>'common_toolbox');
	$PAGE['hmatrix_augment'] = array('title'=>'hmatrix_augment', 'parent'=>'common_toolbox');
	$PAGE['hmatrix_rotate_about'] = array('title'=>'hmatrix_rotate_about', 'parent'=>'common_toolbox');
	$PAGE['hmatrix_rotate_euler'] = array('title'=>'hmatrix_rotate_euler', 'parent'=>'common_toolbox');
	$PAGE['hmatrix_rotate_x'] = array('title'=>'hmatrix_rotate_x', 'parent'=>'common_toolbox');
	$PAGE['hmatrix_rotate_y'] = array('title'=>'hmatrix_rotate_y', 'parent'=>'common_toolbox');
	$PAGE['hmatrix_rotate_z'] = array('title'=>'hmatrix_rotate_z', 'parent'=>'common_toolbox');
	$PAGE['hmatrix_scale'] = array('title'=>'hmatrix_scale', 'parent'=>'common_toolbox');
	$PAGE['hmatrix_scale_get'] = array('title'=>'hmatrix_scale_get', 'parent'=>'common_toolbox');
	$PAGE['hmatrix_translate'] = array('title'=>'hmatrix_translate', 'parent'=>'common_toolbox');
	$PAGE['htransform_planes'] = array('title'=>'htransform_planes', 'parent'=>'common_toolbox');
	$PAGE['htransform_vectors'] = array('title'=>'htransform_vectors', 'parent'=>'common_toolbox');	
	$PAGE['resample_plane'] = array('title'=>'resample_plane', 'parent'=>'common_toolbox');
	$PAGE['reslice_volume'] = array('title'=>'reslice_volume', 'parent'=>'common_toolbox');
	$PAGE['generate_slice_coordinates'] = array('title'=>'generate_slice_coordinates', 'parent'=>'common_toolbox');
	$PAGE['generate_image_coordinates'] = array('title'=>'generate_image_coordinates', 'parent'=>'common_toolbox');
	$PAGE['largest_component'] = array('title'=>'largest_component', 'parent'=>'common_toolbox');
	$PAGE['outside_mask3'] = array('title'=>'outside_mask3', 'parent'=>'common_toolbox');
	
	$PAGE['common_toolbox_other'] = array('title'=>'Common toolbox: Other', 'parent'=>'root');
	$PAGE['iff'] = array('title'=>'iff', 'parent'=>'common_toolbox_other');
	$PAGE['inimanage'] = array('title'=>'inimanage', 'parent'=>'common_toolbox_other');
	$PAGE['safeget'] = array('title'=>'safeget', 'parent'=>'common_toolbox_other');	

	$PAGE['epri_create_directory'] = array('title'=>'epri_create_directory', 'parent'=>'common_toolbox_other');	
		
	//---- EPRI TOOLBOX -----------------------------------------------------
	$PAGE['epri_toolbox'] = array('title'=>'EPRI toolbox', 'parent'=>'root');
	$PAGE['epri_msps'] = array('title'=>'epri_msps', 'parent'=>'epri_toolbox');
	$PAGE['epri_msps_decrypt'] = array('title'=>'epri_msps_decrypt', 'parent'=>'epri_toolbox');
	$PAGE['epri_navigator'] = array('title'=>'epri_navigator', 'parent'=>'epri_toolbox');
	$PAGE['epri_navigator_split'] = array('title'=>'epri_navigator_split', 'parent'=>'epri_toolbox');
	$PAGE['epri_baseline'] = array('title'=>'epri_baseline', 'parent'=>'epri_toolbox');
	$PAGE['epri_baseline_split'] = array('title'=>'epri_baseline_split', 'parent'=>'epri_toolbox');
	$PAGE['epri_split_field'] = array('title'=>'epri_split_field', 'parent'=>'epri_toolbox');
	
	$PAGE['epr_llw_po2'] = array('title'=>'epr_LLW_PO2', 'parent'=>'epri_toolbox');
	$PAGE['epr_t2_po2'] = array('title'=>'epr_T2_PO2', 'parent'=>'epri_toolbox');
	$PAGE['epr_r2_po2'] = array('title'=>'epr_R2_PO2', 'parent'=>'epri_toolbox');
	
	
	//---- IBGUI TOOLBOX -----------------------------------------------------
	$PAGE['ibgui_toolbox'] = array('title'=>'ibGUI toolbox', 'parent'=>'root');
	$PAGE['ibgui'] = array('title'=>'ibGUI', 'parent'=>'ibgui_toolbox');

	//---- ARBIZGUI TOOLBOX -----------------------------------------------------
	$PAGE['arbuzgui_toolbox'] = array('title'=>'ArbuzGUI toolbox', 'parent'=>'root');
	$PAGE['arbuzgui'] = array('title'=>'ArbuzGUI', 'parent'=>'arbuzgui_toolbox');

	//---- PROCESSGUI TOOLBOX -----------------------------------------------------
	$PAGE['processgui_toolbox'] = array('title'=>'ProcessGUI toolbox', 'parent'=>'root');
	$PAGE['processgui'] = array('title'=>'ProcessGUI', 'parent'=>'processgui_toolbox');
	$PAGE['processloadscenario'] = array('title'=>'ProcessLoadScenario', 'parent'=>'processgui_toolbox');	
	$PAGE['processgetdefinition'] = array('title'=>'ProcessGetDefinition', 'parent'=>'processgui_toolbox');	
	$PAGE['processformcontroller'] = array('title'=>'ProcessFormController', 'parent'=>'processgui_toolbox');	
	$PAGE['processvaluedialogdlg'] = array('title'=>'ProcessValueDialogDLG', 'parent'=>'processgui_toolbox');
		
	//---- USEFUL UTILITIES -----------------------------------------------------
	$PAGE['utilities'] = array('title'=>'Useful utilities', 'parent'=>'root');
	$PAGE['matrixgui'] = array('title'=>'MatrixGUI', 'parent'=>'utilities');
	$PAGE['load_bruker_image'] = array('title'=>"Load Bruker data", 'parent'=>'utilities');
	
	$PAGE['used_toolboxes'] = array('title'=>'Toolboxes used', 'parent'=>'root');
	
	//---- HELP the helpers -----------------------------------------------------
	$PAGE['helpmanual'] = array('title'=>'This help', 'parent'=>'root');
	
	
	
	// Verification of links
	foreach($PAGE as $key=>$value)
	{
		if($value['parent'] === 'root') continue;
		if(array_key_exists($value['parent'], $PAGE)) continue;
		echo('structure: key "'.$value['parent'].'" referenced by "'.$key.'" does not exist'."\n");
	
	}
	
	foreach($PAGE as $key=>$value)
	{
		if($value['parent'] === 'root')
			$PAGE[$key]['level']=0;
		else
		{
			$parent_level = $PAGE[$value['parent']]['level'];
			$PAGE[$key]['level']=$parent_level + 1;
		}
	}	
?>