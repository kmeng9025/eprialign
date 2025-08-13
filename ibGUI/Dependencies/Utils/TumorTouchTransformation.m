% A = TumorTouchTransformation(TumorEllipsoid, m_dim_voxels, m_size_cm)
% Boris 
function A = TumorTouchTransformation(TumorEllipsoid, m_dim_voxels, m_size_cm)

expInfo = safeget(TumorEllipsoid, 'expInfo', []);
rGap      = safeget(expInfo, 'rGap', [1,1,1]);
rOffset      = safeget(expInfo, 'rOffset', [0,0,0]);
magnetType = safeget(expInfo, 'magnetType', 1);
resonType = safeget(expInfo, 'resonType', '');

% transform to coordinate system
Astereo2gap = hmatrix_translate(-rGap);
sizeReson = epr_GetResonatorSize(resonType);
Dcm=sizeReson(1); Lcm=sizeReson(2);
Ares_center2grad = hmatrix_translate(rOffset);

switch magnetType
  case 0, %default image
    Agap2res_center  = hmatrix_translate([-Lcm/2 Dcm/2 0]);
  case 1, % small magnet
    AflipZ = hmatrix_scale([1,1,-1]);
    Agap2res_center  = hmatrix_translate([0, -Lcm/2, -Dcm/2])*AflipZ;
  otherwise % intermediate magnet
   AflipXY    = hmatrix_rotate_z(90)*hmatrix_scale([1,-1,1]);
    Agap2res_center  = hmatrix_translate([Dcm/2 -Lcm/2 0]);
end
A = Astereo2gap*Agap2res_center*Ares_center2grad;

if exist('m_size_cm', 'var') && exist('m_dim_voxels', 'var')
  if numel(find(m_dim_voxels > 1)) == 2
    the_scale = m_size_cm(1:2)./(m_dim_voxels(1:2)-1);
    the_scale(3) = 1;
    the_shift = -(m_dim_voxels+1)/2; the_shift(3) = 0;
    pix2size = hmatrix_translate(the_shift)*hmatrix_scale(the_scale);
  else
    pix2size = hmatrix_translate(-(m_dim_voxels+1)/2)*hmatrix_scale(m_size_cm./(m_dim_voxels-1));
  end
  A = A * inv(pix2size);
end

