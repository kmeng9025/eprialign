 <?
/**
 * Multybyte version of ucfirst
 * @param string $str
 * @return string
 */
function fa_mb_ucfirst($str) {
$fc = mb_strtoupper(mb_substr($str, 0, 1, 'UTF-8'), 'UTF-8');
return $fc.mb_substr($str, 1, 1000, 'UTF-8');
}

/**
 * Generate HTML code for reference
 * @param string $ref reference
 * @param string $text reference text
 * @param string $add_par parameters of <a>
 * @return string
 */
function fa_afield($ref,$text,$add_par='')
	{
		if ($add_par != '') $add_par = ' '.$add_par;
		return '<a href=\''.$ref.'\''.$add_par.'>'.$text.'</a>';
	}
	
/**
 * Generate HTML code for image
 * @param string $fname image filename
 * @param int $width image width
 * @param int $height image height
 * @param string $pars parameters of <img>
 * @return string
 */
function fa_imgfield($fname, $width=0, $height=0, $pars='')
	{
		if($width) $pars = $pars.' width = "'.(int)$width.'"';
		if($height) $pars = $pars.' height = "'.(int)$height.'"';
		return '<img src = "'.$fname.'" '.$pars.' />';
	}	
	
function fa_div_pars($left, $top, $width, $height, $zindex, $add_pars)
{
	return sprintf('style="position:absolute; top:%dpx; left:%dpx; width:%dpx; height:%dpx; z-index:%d; %s"', 
		round($top), round($left), round($width), round($height), $zindex, $add_pars);
}
  
define('FA_SEARCH_EXACT', 0);
define('FA_SEARCH_CONTAINS', 1);
/**
 * Search array for values of one of the elements
 * @param array $input
 * @param string $subkey
 * @param unknown_type $value
 * @return array of search results where 'search_key' element is added
 */
function search_by_subkey($input, $subkey, $value, $search_condition = FA_SEARCH_EXACT)
{
	$res = array();
	if($search_condition == FA_SEARCH_EXACT)
		{
		foreach($input as $key => $inp)
			if($inp[$subkey] == $value)
			{	
				$inp['search_key'] = $key;
				$res[$key] = $inp;
			}
		}	
	elseif($search_condition == FA_SEARCH_CONTAINS)
		foreach($input as $key => $inp)
			if(strpos($inp[$subkey], $value) !==false)
			{	
				$inp['search_key'] = $key;
				$res[$key] = $inp;
			}	
	return $res;
}	

define('DTF_DD_month_YYYY', 0);
define('DTF_DD_MM_YYYY', 1);
define('DTF_YYYY', 2);
define('DTF_month_YYYY', 3);
/**
 * Print date in varous formats
 * @param array(0,1,2) $date
 * @param int $the_format [DTF_DD_month_YYYY/DTF_DD_MM_YYYY]
 * @param string $the_lang
 * @return string
 */
function fa_print_date($date, $the_format = DTF_DD_month_YYYY, $the_lang)
{
	global $WORD;
	
	$the_date = '';
	if (count($date) == 0) return '';
	switch ($the_format)
	{
  		case DTF_DD_MM_YYYY:
 /*   		if vec(1), $the_date = sprintf('%4d',vec(1)); end
    		if vec(2), $the_date = [sprintf('%2d',vec(2)),'.',$the_date]; end
    		if vec(3), $the_date = [sprintf('%2d',vec(3)),'.',$the_date]; end
    		$the_date($the_date == ' ') = '0';*/
    		break;
/*  		case 'MM.DD.YYYY':
    		if vec(2), $the_date = sprintf('%2d',vec(2)); end
    		if isempty($the_date), conn=''; else conn='.'; end
    		if vec(3), $the_date = [$the_date, sprintf('%s%2d',conn,vec(3))]; end
    		if isempty($the_date), conn=''; else conn='.'; end
    		if vec(1), $the_date = [$the_date, sprintf('%s%4d',conn,vec(1))]; end
    		$the_date($the_date == ' ') = '0';*/
  		case DTF_YYYY:
			if (!is_null($date[0]))
		          $the_date = sprintf('%d', $date[0]);	
		    break;
/*  		case 'DD MON YYYY':
		    months = {'JAN','FEB','MAR','APR','MAY','JUN', ...
		      'JUL','AUG','SEP','OCT','NOV','DEC'};
		    if vec(1), $the_date = sprintf('%4d',vec(1)); end
		    if vec(2), $the_date = sprintf('%s %s',months{vec(2)},$the_date); end
		    if vec(3), $the_date = sprintf('%2d %s',vec(3),$the_date); end*/
  		case DTF_month_YYYY:
		    if ($date[0] == 0) return '=';
		    if (!is_null($date[0]) && !is_null($date[1]))
		          $the_date = sprintf('%s %d %s',
		            fa_get_word($WORD['first_month']+$date[1]-1, $the_lang, 'i'),$date[0],
		            '');	 
		    elseif (!is_null($date[0]))
		          $the_date = sprintf('%d', $date[0]);	
		    break;
  		case DTF_DD_month_YYYY:
		    if ($date[0] == 0) return '=';
		    if (!is_null($date[0]) && !is_null($date[1]) && !is_null($date[2]))
		          $the_date = sprintf('%d %s %d %s', $date[2],
		            fa_get_word($WORD['first_month']+$date[1]-1, $the_lang, 'r'),$date[0],
		            $the_lang == 'rus' ? fa_get_word($WORD['date_year'], $the_lang, 'r') : '');		    	
		    elseif (!is_null($date[0]) && !is_null($date[1]))
		          $the_date = sprintf('%s %d %s',
		            fa_get_word($WORD['first_month']+$date[1]-1, $the_lang, 'i'),$date[0],
		            ($the_lang == 'rus' ? fa_get_word($WORD['date_year'], $the_lang, 'r') : ''));	 
		    elseif (!is_null($date[0]))
		          $the_date = sprintf('%d', $date[0]);	
		    break;
	}
	return $the_date;
}

function fa_photo_date($the_range_field, $the_year_field, $the_range)
{
	global $WORD;
	
	$year_range_eng = ''; $year_range_rus = ''; $year_increment = 0;
	$year_only = intval($the_year_field);
	switch($the_range_field)
		{
			case 'R' :
				$d1 = $the_range[0];
				$d2 = $the_range[1];
				$year_range_rus = $d1.' - '.
					$d2.'-'.fa_get_word($WORD['year_ending'],'rus', 'i', 'P', 'c').' '.
					fa_get_word($WORD['date_year'],'rus', 'i', 'P', 'c');
				$year_range_eng = $d1.' - '.
					$d2.fa_get_word($WORD['year_ending'],'eng', 'i', 'P', 'c');
				break;
			case 'D' :
				$year_range_eng = $year_only.'s';
				$year_range_rus = $year_only.
					'-'.fa_get_word($WORD['year_ending'],'rus', 'i', 'P', 'c').' '.fa_get_word($WORD['date_year'],'rus', 'i', 'P', 'c');
				$year_increment = 5;
				break;
			case 'E' :
				$year_range_eng = fa_get_word($WORD['year_early'],'eng', 'i', '', 'Cc').' '.$year_only.'s';
				$year_range_rus = fa_get_word($WORD['year_early'],'rus', 'i', '', 'Cc').' '.$year_only.'-'.
					fa_get_word($WORD['year_ending'],'rus', 'r', 'P', 'c').' '.
					fa_get_word($WORD['date_year'],'rus', 'r', 'P', 'c');
				$year_increment = 2;
				break;
			case 'L' :
				$year_range_eng = fa_get_word($WORD['year_late'],'eng', 'i', '', 'Cc').' '.$year_only.'s';
				$year_range_rus = fa_get_word($WORD['year_late'],'rus', 'i', '', 'Cc').' '.$year_only.'-'.
					fa_get_word($WORD['year_ending'],'rus', 'r', 'P', 'c').' '.
					fa_get_word($WORD['date_year'],'rus', 'r', 'P', 'c');
				$year_increment = 7;
				break;
			case 'M' :
				$year_range_eng = fa_get_word($WORD['year_mid'],'eng', 'i', '', 'Cc').' '.$year_only.'s';
				$year_range_rus = fa_get_word($WORD['year_mid'],'rus', 'i', '', 'Cc').' '.$year_only.
					'-'.fa_get_word($WORD['year_ending'],'rus', 'r', 'P', 'c').' '.fa_get_word($WORD['date_year'],'rus', 'r', 'P', 'c');
				$year_increment = 5;
				break;
			case 'A' :
				$year_range_rus = fa_get_word($WORD['year_late'],'rus', 'i', '', 'Cc').' '.
				($year_only-10).'-'.fa_get_word($WORD['year_ending'],'rus', 'r', 'P', 'c').' - '.
					fa_get_word($WORD['year_early'],'rus', 'i', '', 'cc').' '.
					$year_only.'-'.fa_get_word($WORD['year_ending'],'rus', 'r', 'P', 'c').' '.
					fa_get_word($WORD['date_year'],'rus', 'r', 'P', 'c');
				$year_range_eng = fa_get_word($WORD['year_late'],'eng', 'i', '', 'Cc').' '.
				($year_only-10).fa_get_word($WORD['year_ending'],'eng', 'r', 'P', 'c').' - '.
					fa_get_word($WORD['year_early'],'eng', 'i', '', 'cc').' '.
					$year_only.fa_get_word($WORD['year_ending'],'eng', 'r', 'P', 'c');
				break;
			case '~' :
				$year_range_eng = fa_get_word($WORD['year_approx'],'eng', 'i', '', 'Cc').' '.$year_only;
				$year_range_rus = fa_get_word($WORD['year_approx'],'rus', 'i', '', 'Cc').' '.$year_only.
				'-'.fa_get_word($WORD['year_ending'],'rus', 'i').' '.fa_get_word($WORD['date_year'],'rus', 'i', '', 'c');
				break;
			default :
				// Just an year
				if($the_year_field - $year_only < 0.00005)
				{
					$year_range_eng = $year_only;
					$year_range_rus = $year_only.' '.fa_get_word($WORD['date_year'],'rus', 'i', '', 'c');
				}
				else
				{
					$month_only = intval(($the_year_field - $year_only)*100+0.001);
					$day_only = intval(($the_year_field - $year_only - $month_only*0.01)*10000 + 0.001);
					// echo($the_year_field." ==== ".$year_only." == ".$month_only." == ".$day_only." == \n");
					
					if($day_only == 0 && $month_only != 0)
					{
						$year_range_eng = fa_get_word($WORD['first_month']+$month_only-1, 'eng', 'i', '', 'Cc').' '.$year_only;
						$year_range_rus = fa_get_word($WORD['first_month']+$month_only-1, 'rus', 'i', '', 'Cc').' '.
							$year_only.' '.fa_get_word($WORD['date_year'],'rus', 'r', '', 'c');
					}
//					elseif($day_only != 0 && $month_only != 0)
					else
					{
						$year_range_eng = fa_get_word($WORD['first_month']+$month_only-1, 'eng', 'i', '', 'Cc').' '.$year_only;
						$year_range_rus = fa_get_word($WORD['first_month']+$month_only-1, 'rus', 'i', '', 'Cc').' '.
							$year_only.' '.fa_get_word($WORD['date_year'],'rus', 'r', '', 'c');
					}
				}
		}
		
	return array($year_range_eng, $year_range_rus, $year_increment);	
}
	
/**
 * Insert HTML references instead of IDs. The key function for hypertext link of the documents.
 * @param array_ref $mentioned lists all references. 
 * 	Return fields: people, docs, refs
 * @param string $the_text text with IDs
 * @param string $the_lang [rus/eng]
 * @param id-string $the_self_ref
 * @param int $the_level [0/1] 0-for root, 1-for subdirectories
 * @param string $the_given_style
 * @param string $the_style_self_ref
 * @return string HTML code
 * */
function sm_insert_reference(&$mentioned, $the_text, $the_self_ref, $the_level)
{
	global $PAGE;
	global $FOLDER;
	
	$mentioned['pages'] = array();
	
	// find all items
	$res = array(); $res1 = array();
	
	// match all $IDs$ with bracket parameters $ID[bla]$
	preg_match_all('/\$(?P<id>[\w\?]+)(?P<arg>\[[\w\s\,]+\]){0,1}\$/i', $the_text, $res);

	$docrefid = 1;
	$replace_mode = true;
	$example_mode = false;
	
	for($i=0; $i < count($res['id']); $i++)
	{
	    $reftext = $res[0][$i];
		$refkey = $res['id'][$i];
		$refarg = $res['arg'][$i];
		
		// Parse replacement mode keys
	    if(strpos($refkey, 'STOP')!== false)
			{
				$replace_mode = false;
				echo ('fa_insert_reference: Reference mode: OFF'."\n");
	    		$the_text = str_replace($reftext, '', $the_text);
				continue;
			}
		elseif(strpos($refkey, 'START')!== false)
			{
				$replace_mode = true;
				echo ('fa_insert_reference: Reference mode: ON'."\n");
	    		$the_text = str_replace($reftext, '', $the_text);
	    		continue;
			}
	    elseif(strpos($refkey, 'EXAMON')!== false)
			{
				$example_mode = true;
				echo ('fa_insert_reference: Example mode: OFF'."\n");
	    		$the_text = str_replace($reftext, '', $the_text);
				continue;
			}
		elseif(strpos($refkey, 'EXAMOFF')!== false)
			{
				$example_mode = false;
				echo ('fa_insert_reference: Example mode: ON'."\n");
	    		$the_text = str_replace($reftext, '', $the_text);
	    		continue;
			}
	    
		if ($replace_mode == false) continue;

		/*
		$padezh_keyword = 'i';
		if($refarg != '')
		{
			$padezh_symbol = $refarg[strlen($refarg)-2]; // this is string of the type [....PADEZH]
			$pos = strpos('rdtvpRDTVPC', $padezh_symbol);
			if($pos !== false)
				$padezh_keyword = $padezh_symbol;
//			echo($reftext.' =>'.$refkey.','.$refarg.'=>'.$padezh_symbol."\n");
		}		
		*/
		
		// check if ID refers to people
		if(array_key_exists($refkey, $PAGE))
	    {
			$mentioned['pages'][] = $refkey; 

				/*
	    	// convert arguments into styles
			$the_style = $the_given_style;
			if($refarg != '')
			{
				if (strpos($refarg, 'iofdf') !== false)
					$the_style = 'name_patronimic_family_nee';
			  	elseif (strpos($refarg, 'iof') !== false)
					$the_style = 'name_patronimic_family';
			  	elseif (strpos($refarg, 'idf') !== false)
					$the_style = 'name_nee';
			  	elseif (strpos($refarg, 'if') !== false)
					$the_style = 'name_family';
			  	elseif (strpos($refarg, 'io') !== false)
					$the_style = 'name_patronimic';
			  	elseif (strpos($refarg, 'i') !== false)
					$the_style = 'name';
			}
			*/
			
			$the_link = fa_afield($refkey.'.html', $PAGE[$refkey]['title']);
	    	$the_text = str_replace($reftext, $the_link, $the_text);
	    }
	    else 
	    	echo 'sm_insert_reference: unknown object with ID = '.$res['id'][$i].' is found. '."\n";
	    
	  }

	// Leave only unique references
	$mentioned['people'] = array_unique($mentioned['pages']);
	
	return $the_text;
}

function matlab_color_code($code, $use_example)
{
	global $PAGE;
	
	$code_split = explode("\n", $code);
	$new_code = array();
	foreach($code_split as $line)
	{
		if(strpos($line, 'exon')) continue;
		if(strpos($line, 'exoff')) continue;
		if(strpos($line, 'codeon')) continue;
		if(strpos($line, 'codeoff')) continue;
		
		$line = rtrim($line);
		
		// comments
		$pos = strpos($line, '%');
		$comment = '';
		if($pos !== false)
		{
			$comment = '<span class="matlab_comment">'.substr($line, $pos).'</span>';
			$code_part = substr($line, 0, $pos);
		}
		else
			$code_part = $line;
		
		preg_match_all('/(?<func>\w+)\s*\([^\(\)]*\)/', $code_part, $matches);
		for ($i = 0; $i < count($matches['func']); $i++) 
		{
			$search = $matches['func'][$i];
			$function_name = strtolower($search);
			if(array_key_exists($function_name, $PAGE))
			{
				$code_part = str_replace($search, '<a href="'.$function_name.'.html" class="eprit_function">'.$search.'</a>',$code_part);
			}
			else
				$code_part = str_replace($search, '<span class="matlab_function">'.$search.'</span>',$code_part);
			// print_r($search);
		}

		preg_match_all('/[\'](?<func>[^\'\"]+)[\']/', $code_part, $matches);
		for ($i = 0; $i < count($matches['func']); $i++) 
		{
			$search = $matches[0][$i];
			$code_part = str_replace($search, '<span class="matlab_literal">'.$search.'</span>',$code_part);
			//print_r($matches);
		}

		preg_match_all('/[^\w](?<func>[\-\+\d\.\e]+)/', $code_part, $matches);
		for ($i = 0; $i < count($matches['func']); $i++) 
		{
			$full_match = str_replace($matches['func'][$i], '<span class="matlab_value">'.$matches['func'][$i].'</span>',$matches[0][$i]);
			$code_part = str_replace($matches[0][$i], $full_match, $code_part);
			//print_r($matches);
		}
		
/*		// ppl operators
		$ops = array();
		$ops[] = 'parallel'; $ops[] = 'if'; $ops[] = 'detect'; $ops[] = 'time';
		$ops[] = 'end'; $ops[] = 'else'; $ops[] = 'wait'; $ops[] = 'int'; $ops[] = 'signal';
		foreach($ops as $op)
		{
			$a1 = strpos($code_part, $op);
			if($a1 !== false)
				$code_part = substr_replace($code_part, '<span class="ppl_keyword">'.$op.'</span>', $a1, strlen($op));
		}
		
 		// ppl operators
		$ops = array();
		$ops[] = 'mwpulse'; 
		$ops[] = 'rfpulse'; 
		foreach($ops as $op)
		{
			$a1 = strpos($code_part, $op);
			if($a1 !== false)
				$code_part = substr_replace($code_part, '<span class="ppl_user">'.$op.'</span>', $a1, strlen($op));
		} */

		$new_code[]=$code_part.$comment;
	}
	$example = '<p class="matlab_example_caption">Example:</p>'."\n";
	$legend = '<p class="matlab_example_legend">Legend: <span class="eprit_function">EPR-IT functions</span>; <span class="matlab_function">MATLAB functions</span>; <span class="matlab_comment">comments</span>.</p>'."\n";
	if($use_example)
	   return $example.'<p class="matlab_code">'.implode("\n", $new_code).'</p>'."\n".$legend;
   
	return '<p class="matlab_code">'.implode("\n", $new_code).'</p>'."\n";   
}
	
function color_syntax($code)
{
	$code_split = explode("\n", $code);
	$new_code = array();
	foreach($code_split as $line)
	{
		if(strpos($line, 'syntaxon')) continue;
		if(strpos($line, 'syntaxoff')) continue;
		$line = rtrim($line);
			
	
		preg_match('/(?<func>\w+)\s*\([^\(\)]*\)/', $line, $matches, PREG_OFFSET_CAPTURE, 0);
		if(!empty($matches)) 
		{
			$search = $matches['func'][0];
			$line = str_replace($search, '<b>'.$search.'</b>',$line);
			// print_r($search);
		}

		$new_code[]=$line;
	}
		
	$example = '<p class="matlab_syntax_caption">Syntax:</p>'."\n";
	return $example.'<p class="matlab_syntax">'.implode("\n", $new_code).'</p>'."\n";
}	

function color_description($code)
{
	$code_split = explode("\n", $code);
	$new_code = array();
	foreach($code_split as $line)
	{
		if(strpos($line, 'descon')) continue;
		if(strpos($line, 'descoff')) continue;
		$line = rtrim($line);
			
		preg_match('/\~\.(?<func>\w+)/', $line, $matches, PREG_OFFSET_CAPTURE, 0);
		if(!empty($matches)) 
		{
			$search = $matches['func'][0];
			$line = replace_first($search,'<span class="structure_field">'.$search.'</span>',$line); 
		}
		else
		{
			preg_match('/(?<func>\w+)\s+\-/', $line, $matches, PREG_OFFSET_CAPTURE, 0);
			if(!empty($matches)) 
			{
				$search = $matches['func'][0];
				$line = replace_first($search,'<span class="structure_field">'.$search.'</span>',$line); 
			}
		}

		preg_match_all('/[\'](?<func>[^\'\"]+)[\']/', $line, $matches);
		for ($i = 0; $i < count($matches['func']); $i++) 
		{
			$search = $matches[0][$i];
			$line = str_replace($search, '<span class="description_literal">'.$search.'</span>',$line);
			//print_r($matches);
		}

		$new_code[]=$line;
	}
	$example = '<p class="matlab_desc_caption">Description:</p>'."\n";
	return $example.'<p class="matlab_description">'.implode("\n", $new_code).'</p>'."\n";
}	

function set_see_also($code)
{
	$code_split = explode("\n", $code);
	$new_code = '';
	$separator = '';
	foreach($code_split as $line)
	{
		if(strpos($line, 'seeon')) continue;
		if(strpos($line, 'seeoff')) continue;
		$line = rtrim($line);
		
		$new_code=$new_code.$separator.'$'.strtolower($line).'$';
		$separator = ', ';
	}
	$example = '<p class="see_also">See also:</p>'."\n";
	return $example.'<p class="see_also_list">'.$new_code.'</p>'."\n";
}	
		
function replace_first($find, $replace, $subject) 
{
    // stolen from the comments at PHP.net/str_replace
    // Splits $subject into an array of 2 items by $find,
    // and then joins the array with $replace
    return implode($replace, explode($find, $subject, 2));
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
function getTime()
	{
	$a = explode (' ',microtime());
	return(double) $a[0] + $a[1];
	}
?>