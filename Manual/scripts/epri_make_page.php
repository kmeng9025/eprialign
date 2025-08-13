<?
function epri_make_page($page)
{
	global $PAGE;
	global $FOLDER;
	
	$eol = "\n";
	$page_text = file_get_contents("template_page.html");
	
	$page_text = str_replace('%%HEAD%%', '', $page_text);
	$page_text = str_replace('%%MENU%%', 'EPR Imaging Toolbox collection user manual', $page_text);
	
	// Build navigation
	$navigation = '<table width="100%" class="page_navigation">'.$eol.'<tr><td class="menu_group0">&nbsp;</td></tr>'.$eol;
	foreach($PAGE as $key=>$navpage)
	{
		$class='';

		if($navpage['level']==0) ;
		elseif(nav_the_same_tree($key, $page)) ;
		elseif(nav_the_same_group($key, $page)) ;
		elseif(nav_children_group($key, $page)) ;
		else continue;

		$class='menu_group'.$PAGE[$key]['level'];
		if($key === $page) $class = $class.' menu_selected';
		
		$file_name = $key.'.html';
		
		$navigation = $navigation.'<tr><td class="'.$class.'">'.fa_afield($file_name, $navpage['title']).'</td></tr>'.$eol;
	}
	$navigation = $navigation."</table>\n";
	$page_text = str_replace('%%NAVIGATION%%', $navigation, $page_text);

	$page_content = 'Missing content';
	$mentioned = array();
	if(strpos($page, '*')===false && file_exists($FOLDER['content'].$page.'.txt'))
	{
		$page_content = file_get_contents($FOLDER['content'].$page.'.txt');
		
		$page_content = '<p class="func_name">'.$PAGE[$page]['title'].'</p><hr>'.$page_content;
		
		do
		{
		$a1 = strpos($page_content, '$exon$');
		$a2 = strpos($page_content, '$exoff$');
		
		if($a1 === false || $a2 ===false) break;

		$ppl_code = matlab_color_code(substr($page_content, $a1, $a2-$a1+strlen('$exoff$')), true);
		$page_content = substr_replace($page_content, $ppl_code, $a1, $a2-$a1+strlen('$exoff$'));
		} while(1);

		do
		{
		$a1 = strpos($page_content, '$codeon$');
		$a2 = strpos($page_content, '$codeoff$');
		
		if($a1 === false || $a2 ===false) break;

		$ppl_code = matlab_color_code(substr($page_content, $a1, $a2-$a1+strlen('$codeoff$')), false);
		$page_content = substr_replace($page_content, $ppl_code, $a1, $a2-$a1+strlen('$codeoff$'));
		} while(1);

		
		do
		{
		$a1 = strpos($page_content, '$syntaxon$');
		$a2 = strpos($page_content, '$syntaxoff$');
		
		if($a1 === false || $a2 ===false) break;

		$ppl_code = color_syntax(substr($page_content, $a1, $a2-$a1+strlen('$syntaxoff$')));
		$page_content = substr_replace($page_content, $ppl_code, $a1, $a2-$a1+strlen('$syntaxoff$'));
		} while(1);

		do
		{
		$a1 = strpos($page_content, '$seeon$');
		$a2 = strpos($page_content, '$seeoff$');
		
		if($a1 === false || $a2 ===false) break;

		$ppl_code = set_see_also(substr($page_content, $a1, $a2-$a1+strlen('$seeoff$')));
		$page_content = substr_replace($page_content, $ppl_code, $a1, $a2-$a1+strlen('$seeoff$'));
		} while(1);

		do
		{
		$a1 = strpos($page_content, '$descon$');
		$a2 = strpos($page_content, '$descoff$');
		
		if($a1 === false || $a2 ===false) break;

		$ppl_code = color_description(substr($page_content, $a1, $a2-$a1+strlen('$descoff$')));
		$page_content = substr_replace($page_content, $ppl_code, $a1, $a2-$a1+strlen('$descoff$'));
		} while(1);

		$page_content = sm_insert_reference($mentioned, $page_content, '', '');
	}
	$page_text = str_replace('%%CONTENT%%', $page_content, $page_text);
	
	// Save
	$efid = fopen($FOLDER['root'].$page.'.html','w');
	fwrite($efid,$page_text);
	fclose($efid);		
}	

function nav_the_same_tree($item, $current_page)
{
	global $PAGE;
	
	$mpage = $current_page;
	while($item != $mpage)
	{
		if($PAGE[$mpage]['level'] == 0) return false;
		$mpage = $PAGE[$mpage]['parent'];
	}
	
	return true;
}

function nav_the_same_group($item, $current_page)
{
	global $PAGE;
	
	if($PAGE[$current_page]['level'] == 0) return false;

	$parent = $PAGE[$current_page]['parent'];
	
	if($PAGE[$item]['parent'] == $PAGE[$current_page]['parent']) return true;	
	return false;	
}

function nav_children_group($item, $current_page)
{
	global $PAGE;
	
	$children = search_by_subkey($PAGE, 'parent', $current_page);
	
	foreach($children as $key=>$value)
	{
		if($key == $item) return true;
	}
	return false;
}

?>