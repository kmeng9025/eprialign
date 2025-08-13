<?
	// Load data
	global $PATHLIST;
	global $PAGE;
	
	include_once('epri_utils.php');
	include_once('epri_structure.php');
	include_once('epri_make_page.php');
	include_once('epri_list.php');
	
	$n = total_number();
    echo('Total number of files: '.$n."\n");
	echo('Included files: '.count($PAGE)."\n");
	
	
	$Start = getTime();	 
	
	foreach($PAGE as $key=>$page)
	{
		epri_make_page($key);
	}
	
	
	echo "Time taken for generation = ".number_format((getTime() - $Start),2)." secs\n";

	$i=0; $j=0;
	foreach ($PATHLIST as $key=>$toolbox)
		foreach($toolbox['members'] as $member)
			{
				$file = basename($member, ".m");
				if(!array_key_exists(strtolower($file), $PAGE))
				{
					$i = $i + 1;
					echo($i.': '.$member."\n");
				}
				else
					$j=$j + 1;
			}
	echo $j." done. ".$i." to go.\n";				
?>