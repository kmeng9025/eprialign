%%
clc
hGUI = 0;
arbuz_InitializeProject(hGUI)

new_image.Name      = 'Image1';
new_image.ImageType = '3D';
new_image.Anative   = eye(4);
new_image.Selected  = 0;
new_image.Visible   = 0;
new_image.FileName  = '';
new_image.isLoaded  = 0;
new_image.Slave     = {};

arbuz_AddImage(hGUI, new_image);

new_image.Name      = 'Image2';
arbuz_AddImage(hGUI, new_image);

% arbuz_DeleteImage(hGUI, 'Image1');

new_image.Name      = 'S1';
new_image.ImageType = 'XYZ';
arbuz_AddImage(hGUI, new_image, 'Image1');

new_image.Name      = 'S2';
new_image.ImageType = 'XYZ';
arbuz_AddImage(hGUI, new_image, 'Image1');

arbuz_AddImage(hGUI, new_image, 'Image1');

new_image.Name      = 'S3';
new_image.ImageType = 'XYZ';
arbuz_AddImage(hGUI, new_image, 'Image1');

arbuz_DeleteImage(hGUI, {'Image1', 'S3'});

%%
output_list = arbuz_FindImage(hGUI, 'master', '', '', {'A', 'slavelist'})
output_list{2}

%%
res = arbuz_get(hGUI, 'FileName')