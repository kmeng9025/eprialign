the_path = 'z:\CenterMATLAB\Images\';
fname = [the_path, 'ico.mat'];
s1 = load(fname);
s1.SliceMaskEdit = imread([the_path, 'SliceMaskEdit.bmp']);
s1.RotateImage = imread([the_path, 'RotateImage.bmp']);
s1.CreateCoordinates = imread([the_path, 'CreateCoordinates.bmp']);
s1.MaskTransfer = imread([the_path, 'MaskTransfer.bmp']);
s1.Information = imread([the_path, 'Information.bmp']);
s1.DefaultViewer = imread([the_path, 'DefaultViewer.bmp']);
s1.CropAndResample = imread([the_path, 'CropAndResample.bmp']);
s1.NotesEdit = imread([the_path, 'NotesEdit.bmp']);
save(fname, '-struct', 's1')

%% i
fname = 'd:\CenterMATLAB\Images\SliceMaskEditICO.mat';
all_images = imread('d:\CenterMATLAB\Images\SliceMaskEditPLG.bmp');

get_one = @(ii,jj) all_images(11 + 30 * ii: 10 + 30 * ii + 24, 11 + 30 * jj: 10 + 30 * jj + 24, :);
ii= 0; jj= 0; s1.AddPoly = get_one(ii,jj);
ii= 0; jj= 1; s1.AddFreeHand = get_one(ii,jj);
ii= 1; jj= 0; s1.RemovePoly = get_one(ii,jj);
ii= 1; jj= 1; s1.FloodFill = get_one(ii,jj);
ii= 2; jj= 0; s1.CircleBrush  = get_one(ii,jj);
ii= 2; jj= 1; s1.CircleEraser = get_one(ii,jj);
ii= 0; jj= 2; s1.CloseImage  = get_one(ii,jj);
ii= 1; jj= 2; s1.OpenImage = get_one(ii,jj);
ii= 0; jj= 3; s1.Propogate = get_one(ii,jj);
ii= 1; jj= 3; s1.Interpolate = get_one(ii,jj);
ii= 1; jj= 4; s1.Pencil1 = get_one(ii,jj);
ii= 2; jj= 2; s1.DeleteAll = get_one(ii,jj);
ii= 2; jj= 3; s1.DeleteContour = get_one(ii,jj);
ii= 0; jj= 4; s1.EraseFreeHand = get_one(ii,jj);
ii= 2; jj= 4; s1.Otsu3D = get_one(ii,jj);

ii= 3; jj= 0; s1.ApplyAll = get_one(ii,jj);
ii= 3; jj= 1; s1.ApplyAllN = get_one(ii,jj);
ii= 3; jj= 2; s1.ApplySlice = get_one(ii,jj);
ii= 3; jj= 3; s1.Largest = get_one(ii,jj);
ii= 3; jj= 4; s1.FillHoles = get_one(ii,jj);

save(fname, '-struct', 's1')
