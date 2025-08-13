function the_new_mask = imrt_tranform_mask(the_mask, destination, pars)

destination = upper(destination);
pix          = safeget(pars, 'PlugPix', 0.025);    % mm
scale        = safeget(pars, 'Plane2PlugScale', 1.26); % mm

switch destination
  case 'BOOST'
    margin       = safeget(pars, 'BoostMargin', 1.2);  % mm
    
    margin_in_plug_coordinates = (margin/scale)+0.2;
    SE = strel('disk',floor(margin_in_plug_coordinates/pix));
    
    the_new_mask = imdilate(the_mask,SE);
  case 'ANTIBOOST'
    margin       = safeget(pars, 'ABoostMargin', 0.6);  % mm
    beam_square  = safeget(pars, 'beam_square', 0);
    
    %     margin_in_plug_coordinates = margin/scale - 0.2;
    margin_in_plug_coordinates = margin*scale;
    if margin_in_plug_coordinates >= 1
      SE = strel('disk',floor(margin_in_plug_coordinates/pix));
      hypoxia_protection = imdilate(the_mask,SE);
      [~,the_new_mask] = expand(imdilate(hypoxia_protection,SE) & ~hypoxia_protection, ~hypoxia_protection, beam_square);
    else
      SE = strel('disk', 1);
      hypoxia_protection = imdilate(the_mask,SE);
      [~,the_new_mask] = expand(imdilate(hypoxia_protection,SE) & ~hypoxia_protection, ~hypoxia_protection, beam_square);
    end
  case 'ANTIBOOST-VER2'
    margin       = safeget(pars, 'ABoostMargin', 0.6);  % mm
    beam_square  = safeget(pars, 'beam_square', 0);
    
    margin_in_plug_coordinates = margin*scale - 0.2;
    SE = strel('disk',floor(margin_in_plug_coordinates/pix));
    SE1 = strel('disk',2*floor(margin_in_plug_coordinates/pix));
    
    slHypo  = the_mask;
    slTumor = pars.slTumor;
    slLEGBEV = imdilate(slTumor,SE1);
    
    hypoxia_protection = imdilate(the_mask,SE);
    
    tumor = slTumor | hypoxia_protection;
    boost = imrt_tranform_mask(slHypo, 'boost', pars);
    
    x = 1:size(tumor,1);
    [X,Y] = meshgrid(x,x);
    
    S = numel(find(boost));
    CM = cm(X,Y,tumor);
    R = sqrt(S/pi);
    ab   = false(size(tumor));
    idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & ~hypoxia_protection & tumor;
    ab(idx_aboost) = true;
    all_protection = tumor & ~hypoxia_protection;
    
    [ismaxedout, ab] = expand(ab, all_protection, S);
    maxS = numel(find(all_protection));
    while ismaxedout && S < maxS
      R = R*1.5;
      idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & all_protection;
      ab(idx_aboost) = true;
      ab = contract(ab, all_protection, S);
      [ismaxedout, ab] = expand(ab, tumor & all_protection, S);
    end
    
    if numel(find(ab)) < S
      R = sqrt(S/pi);
      all_protection = slLEGBEV & ~hypoxia_protection;
      cc = zeros(size(all_protection));
      S1 = numel(find(ab));
      S1prime = S1-1;
      while S1 < S && S1~=S1prime
        ab = ab | cc;
        R = R*1.1;
        idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & all_protection;
        cc(idx_aboost) = true;
        S1prime = S1;
        S1 = numel(find(ab | cc));
      end
    end
    [isupdate, conn] = connect_islands(ab, hypoxia_protection, CM);
    if isupdate, [~,ab] = expand(ab & ~conn, all_protection & ~conn, S); end
    the_new_mask = ab;
  case 'ANTIBOOST-VER3'
    margin       = safeget(pars, 'ABoostMargin', 0.6);  % mm
    beam_square  = safeget(pars, 'beam_square', 0);
    
    margin_in_plug_coordinates = margin*scale - 0.2;
    SE = strel('disk',floor(margin_in_plug_coordinates/pix));
    SE1 = strel('disk',2*floor(margin_in_plug_coordinates/pix));
    
    slHypo  = the_mask;
    slTumor = pars.slTumor;
    slLEGBEV = imdilate(slTumor,SE1);
    
    hypoxia_protection = imdilate(the_mask,SE);
    
    slHypo1 = bwmorph(slHypo, 'bridge');
    se = strel('disk',25);
    slHypo1 = imclose(slHypo1, se);
    
    protection_pars = pars;
    
    boost = imrt_tranform_mask(slHypo, 'boost', pars);
    x = 1:size(slHypo,1);
    [X,Y] = meshgrid(x,x);
    S = numel(find(boost));
%     CM = cm(X,Y,slTumor | MapHypoxia2Protection(slHypo, pars));
    CM = cm(X,Y,slTumor);
    
    ABoostMargin = linspace(pars.ABoostMargin, 0.0, 5);
    for ii=1:length(ABoostMargin)
      if ABoostMargin(ii) > 0
      protection_pars.BoostMargin = ABoostMargin(ii);
      hypoxia_protection = imrt_tranform_mask(slHypo1, 'boost', protection_pars);
      else
        hypoxia_protection = slHypo1;
      end
      ab = slTumor & ~hypoxia_protection;
      if numel(find(ab)) > S, break; end
    end
    
    if numel(find(ab)) < S
      R = sqrt(S/pi);
      all_protection = slLEGBEV & ~hypoxia_protection;
      cc = zeros(size(all_protection));
      S1 = numel(find(ab));
      while S1 < S
        ab = ab | cc;
        R = R*1.1;
        idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & all_protection;
        cc(idx_aboost) = true;
        S1 = numel(find(ab | cc));
      end
    else
      all_protection = slLEGBEV & ~hypoxia_protection;
      ab = contract(ab, all_protection, S);
    end
    
    
    [~,ab] = expand(ab, all_protection, S);
    [isupdate, conn] = connect_islands(ab, hypoxia_protection, CM);
    if isupdate, [~,ab] = expand(ab & ~conn, all_protection & ~conn, S); end
    the_new_mask = ab;
end


% --------------------------------------------------------------------
function [is, mask1] = connect_islands(mask, block_area, CM)
is = true;
x = 1:size(block_area,1);
[X,Y]=meshgrid(x,x);
mask1 = false(size(mask));

% find islands
all = ~mask|block_area;
CC = bwconncomp(all);

for ii=1:length(CC.PixelIdxList)
  mm = false(size(block_area));
  mm(CC.PixelIdxList{ii}) = true;
  if mm(1,1), continue; end
  disp('Connecting islands.');
  CM = [sum(sum(mm.*X)), sum(sum(mm.*Y))]/sum(sum(mm));
  
  % expand island until it hits something
  all_but_one = all & ~mm;
  se1 = strel('disk', 3);
  mm1 = mm;
  while numel(find(all_but_one & mm1)) == 0
    mm1 = imdilate(mm1, se1);
  end
  % TODO
  % here we need to eliminate multiple overlaps
  
  % calculate CM for overlap
  overlap = all_but_one & mm1;
  cm = [sum(sum(overlap.*X)), sum(sum(overlap.*Y))]/sum(sum(overlap));
  
  % draw line from the center outwards
  dx = cm(1)-CM(1);
  dy = cm(2)-CM(2);
  if abs(dx) > abs(dy)
    dy_dx = (cm(2)-CM(2)) / (cm(1)-CM(1));
    for jj=min(CM(1)-cm(1),0):max(CM(1)-cm(1),0)
      mask1(floor(cm(2)+jj*dy_dx), floor(cm(1)+jj))=true;
    end
  else
    dx_dy = (cm(1)-CM(1)) / (cm(2)-CM(2));
    for jj=min(CM(2)-cm(2),0):max(CM(2)-cm(2),0)
      mask1(floor(cm(2)+jj), floor(cm(1)+dx_dy*jj))=true;
    end
  end
end
se = strel('disk',9);
mask1 = imdilate(mask1, se) & mask;

% --------------------------------------------------------------------
function [ismaxedout,mask] = expand(mask, expansion_area, S)
ismaxedout = false;
se = strel('disk',3);
mask1 = mask;

while numel(find(mask)) < S
  mask1 = imdilate(mask1, se) & expansion_area;
  difference = find(mask1 & ~mask);
  if ~any(difference), ismaxedout = true; break; end
  if numel(find(mask1)) > S
    n1 = numel(find(mask1));
    mask1(difference(1:n1-S)) = false;
  end
  mask = mask1;
end

% --------------------------------------------------------------------
function mask = contract(mask, area, S)
se = strel('disk',3);
mask1 = mask;

while numel(find(mask)) > S
  mask1 = imerode(mask1, se) & area;
  difference = find(mask & ~mask1);
  if ~any(difference), break; end
  n1 = numel(find(mask1));
  if n1 > S
    fprintf('n1=%i S=%i\n', n1, S)
    mask1(difference) = false;
  else
    fprintf('n1=%i S=%i\n', n1, S)
    mask1(difference(1:S-n1)) = true;
  end
  mask = mask1;
end

% --------------------------------------------------------------------
function CM = cm(X,Y,mask)
CM = [sum(sum(mask.*X))/sum(sum(mask)), sum(sum(mask.*Y))/sum(sum(mask))];
