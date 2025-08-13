 <?
 
 function total_number()
 {
	 global $PATHLIST;
	 
	 $root = 'z:\\CenterMATLAB\\';
	 
	 $PATHLIST['iradon'] = array('path' => 'iradon', 'use' => true, 'auto' => true);
	 $PATHLIST['ibGUI'] = array('path' => 'ibGUI', 'use' => true, 'auto' => false);
	 $PATHLIST['ibGUI']['members'] = array('0'=>$root.'\\ibGUI\\ibGUI.m');
	 $PATHLIST['Arbuz'] = array('path' => 'Arbuz2.0', 'use' => true, 'auto' => true);
	 $PATHLIST['common'] = array('path' => 'common', 'use' => true, 'auto' => true);
	 $PATHLIST['eprfit'] = array('path' => 'eprfit', 'use' => true, 'auto' => true);
	 $PATHLIST['radon'] = array('path' => 'radon', 'use' => true, 'auto' => true);
	 $PATHLIST['kazan'] = array('path' => 'ibGUI', 'use' => true, 'auto' => false);
	 $PATHLIST['kazan']['members'] = array('0'=>$root.'\\kazan\\kazan.m');
	 $PATHLIST['reports'] = array('path' => 'reports', 'use' => true, 'auto' => true);
	 $PATHLIST['epri'] = array('path' => 'epri', 'use' => true, 'auto' => true);
	 $PATHLIST['rapid_scan'] = array('path' => 'rapid_scan', 'use' => true, 'auto' => true);
	 
	 $PATHLIST['iradon_mstage'] = array('path' => 'iradon_mstage', 'use' => true, 'auto' => false);
	 $PATHLIST['iradon_mstage']['members'] = array('0'=>$root.'\\iradon_mstage\\iradon_d2d_mstage.m');
	 $PATHLIST['iradon_QiaoFBP3D'] = array('path' => 'iradon_QiaoFBP3D', 'use' => true, 'auto' => false);
	 $PATHLIST['iradon_QiaoFBP3D']['members'] = array('0'=>$root.'\\iradon_QiaoFBP3D\\iradon_qiao_fbp3d.m');
	 $PATHLIST['iradon_Tseylin4D'] = array('path' => 'iradon_Tseylin4D', 'use' => true, 'auto' => true);
	 
	 
	 $n = 0;
	 foreach ($PATHLIST as $key=>$toolbox)
	 {
		if($toolbox['auto'])
			{
				
				$filelist = glob($root.$toolbox['path'].'\\*.m');
				$PATHLIST[$key]['members'] = $filelist;
			}
		if (array_key_exists('members', $PATHLIST[$key]))
			$n = $n + count($PATHLIST[$key]['members']);
		else
			echo('key='.$key." has no members.\n");
	}
	 
	 return $n;
}	 
 
 
function get_MATLAB_help($code)
 {
	 $comment = array();
	 foreach($code as $line)
	 {
		$iscomment = strpos($line, '%')!==false ? true : false;
		if(!$iscomment) break;
		if(strpos($line, 'function')!==false && !$iscomment) break;
		$comment[] = $line;
	 }
	 return $comment;
 }

 ?>