function transform_image_bbox_demo(varargin)

if (nargin > 0), 
    a = varargin{1};
else
    [fn,pn]=uigetfile();
    a=imread([pn fn]);
end;
xsize=size(a,2);
ysize=size(a,1);
bbox=[0.3*xsize 0.5*ysize; 0.8*xsize 0.7*ysize];
hmat1=rotate_translate_scale_image_transform1(a, 0,[0 0],[0.25 0.25],bbox)
hmat2=rotate_translate_scale_image_transform1(a, 0,[0 0],[1 1],bbox)
at=transform_image_bbox(a,hmat1,0,bbox);
at1=transform_image_bbox(a,hmat2,0,bbox);
if (size(at,3) > 3), 
    a = a(:,:,1:3);
    at = at(:,:,1:3); 
    at1 = at1(:,:,1:3); 
end
figure, image(a),axis image;
pts = [bbox(1,:); bbox(1,1) bbox(2,2);bbox(2,:);bbox(2,1) bbox(1,2); bbox(1,:)];
hold on
plot(pts(:,1),pts(:,2),'c-');
figure, image(at),axis image
figure, image(at1),axis image