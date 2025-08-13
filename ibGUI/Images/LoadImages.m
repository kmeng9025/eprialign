%% i
fpath = 'z:\CenterMATLAB\ibGUI\Images\';
fname = 'ibToolBoxMask.mat';
all_images = imread(fullfile(fpath, 'SliceMaskEditPLG.bmp'));
ii= 0; jj= 0; s1.AddPoly = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 0; jj= 1; s1.AddFreeHand = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 0; jj= 2; s1.CloseImage3D  = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 0; jj= 3; s1.Propogate = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 0; jj= 4; s1.EraseFreeHand = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 0; jj= 5; s1.LoadFile = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 0; jj= 6; s1.SaveFile = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 0; jj= 7; s1.Undo = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 0; jj= 8; s1.Redo = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);

ii= 1; jj= 0; s1.RemovePoly = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 1; jj= 1; s1.RemoveSpecles = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 1; jj= 2; s1.OpenImage3D = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 1; jj= 3; s1.Interpolate3D = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 1; jj= 4; s1.SaveFileRes = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 1; jj= 5; s1.SaveFileExcel = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 1; jj= 6; s1.SetAll3D = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 1; jj= 7; s1.SetAll = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 1; jj= 8; s1.SelectByThreshold = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);

ii= 2; jj= 0; s1.CircleBrush  = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 2; jj= 1; s1.CircleEraser = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 2; jj= 2; s1.EraseAll = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 2; jj= 3; s1.EraseAll3D = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 2; jj= 4; s1.FillHoles3D = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 2; jj= 5; s1.Otsu = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 2; jj= 6; s1.Otsu3D = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 2; jj= 8; s1.SelectByThreshold3D = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);
ii= 2; jj= 7; s1.Close = all_images(11 + 20 * ii: 10 + 20 * ii + 16, 11 + 20 * jj: 10 + 20 * jj + 16, :);

save(fullfile(fpath, fname), '-struct', 's1')
