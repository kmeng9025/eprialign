function rgb = arbuz_ind2rgb(im, the_colormap_name, the_scale)

switch the_colormap_name
  case 'jet', the_colormap = jet;
  case 'ijet', the_colormap = flip(jet,2);
  case 'HSV', the_colormap = HSV;
  case 'hot', the_colormap = hot;
  case 'cool', the_colormap = cool;
  case 'bone', the_colormap = bone;
  case 'autumn', the_colormap = autumn;
  case 'spring', the_colormap = spring;
  case 'summer', the_colormap = summer;
  case 'winter', the_colormap = winter;
  case 'gray', the_colormap = gray;
  case 'copper', the_colormap = copper;
  case 'pink', the_colormap = pink;
  otherwise, the_colormap = jet;
end

cmap_size = size(the_colormap, 1);

a = min(the_scale);
b = diff(the_scale);
im = fix(cmap_size*(im-a)/b);
rgb = ind2rgb(im, the_colormap);