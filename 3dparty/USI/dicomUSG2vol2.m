function dicomUSG2vol2
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
javax.swing.UIManager.setLookAndFeel('com.sun.java.swing.plaf.nimbus.NimbusLookAndFeel');
format long;

f = figure('Visible','on','Name','Dicom_USG',...
    'units','normalized','outerposition',[0 0 1 1],...
    'Color',[.94 .97 1]);

uicontrol('Style','pushbutton','Parent',f,...
    'units','normalized',...
    'Position',[.625 .95 .13 .04],...
    'String','Load next dicom file',...
    'Callback',@reset_callback);

    function reset_callback(~,~)
        guzik = questdlg(sprintf('Are you sure you want to load a next dicom file?\nYour previous calculations will disappear!'),...
            'Load next dicom file','Yes','No','No');
        drawnow; pause(0.05);
        if strcmp(guzik,'Yes')
            close(gcbf);
            dicomUSG2vol2
            return;
        else
            return;
        end
    end

uicontrol('Style','pushbutton','Parent',f,...
    'units','normalized',...
    'Position',[.265 .95 .13 .04],...
    'String','Choose a dicom file',...
    'Callback',@sciezka1_callback);

uicontrol('Style','pushbutton','Parent',f,...
    'units','normalized',...
    'Position',[.445 .95 .13 .04],...
    'String','Choose a raw data file',...
    'Callback',@sciezka2_callback);

sciezka2 = '';
skalaX = 0;
skalaY = 0;
sciezka3= '';

    function sciezka1_callback(~,~)
        [nazwa,sciezkaaa]=uigetfile('.dcm');
        sciezkaa=fullfile(sciezkaaa,nazwa);
        [sciezkaaa2,nazwa2,~]=fileparts(sciezkaa);
        sciezka2=fullfile(sciezkaaa2,nazwa2);
        if isempty(sciezka3)
        else
            uiresume(gcbf);
        end
    end

    function sciezka2_callback(~,~)
        [nazwa,sciezkaaa]=uigetfile('.xml');
        sciezkaa=fullfile(sciezkaaa,nazwa);
        [sciezkaaa2,nazwa2,~]=fileparts(sciezkaa);
        sciezka3=fullfile(sciezkaaa2,nazwa2);
        if isempty(sciezka2)
        else
            uiresume(gcbf);
        end
    end

% uiwait(f);

fname = 'v:\data\Imagnet_data_16\03\160323\Ultrasound_Fid_Phntm_Vevo2000\2016-03-23-14-17-06.dcm';
        [sciezkaaa2,nazwa2,~]=fileparts(fname);
        sciezka2=fullfile(sciezkaaa2,nazwa2);

sciezka = strcat(sciezka2,'.dcm');
stak = dicomread(sciezka);
parametry = dicominfo(sciezka);

skalaX = parametry.PixelSpacing(1);
skalaY = parametry.PixelSpacing(2);

sciezka3 = strcat(sciezka3,'.xml');
import javax.xml.xpath.*
file = xmlread(sciezka3);
factory = XPathFactory.newInstance;
xpath = factory.newXPath;

expressionnn = xpath.compile('systemParameters/parameter[@name="Series-Name"]/@value');
nazwa = expressionnn.evaluate(file, XPathConstants.NODE);
nazwaa = char(nazwa.getTextContent);
set(f,'NumberTitle','off');
set(f,'Name',nazwaa);

expression = xpath.compile('systemParameters/parameter[@name="Color-Mode/Depth"]/@value');
W1Node = expression.evaluate(file, XPathConstants.NODE);
expressionn = xpath.compile('systemParameters/parameter[@name="Power-Mode/Depth"]/@value');
W11Node = expressionn.evaluate(file, XPathConstants.NODE);

if isempty(W1Node)
    W1= str2double(W11Node.getTextContent);
else
    W1 = str2double(W1Node.getTextContent);
end

expression1 = xpath.compile('systemParameters/parameter[@name="Color-Mode/Depth-Offset"]/@value');
W2Node = expression1.evaluate(file, XPathConstants.NODE);
expression11 = xpath.compile('systemParameters/parameter[@name="Power-Mode/Depth-Offset"]/@value');
W22Node = expression11.evaluate(file, XPathConstants.NODE);

if isempty(W2Node)
    W2= str2double(W22Node.getTextContent);
else
    W2 = str2double(W2Node.getTextContent);
end

W=(W1-W2)/skalaY;

expression2 = xpath.compile('systemParameters/parameter[@name="Color-Mode/Width"]/@value');
DNode = expression2.evaluate(file, XPathConstants.NODE);
expression22 = xpath.compile('systemParameters/parameter[@name="Power-Mode/Width"]/@value');
D1Node = expression22.evaluate(file, XPathConstants.NODE);

if isempty(DNode)
    D = str2double(D1Node.getTextContent)/skalaX;
else
    D = str2double(DNode.getTextContent)/skalaX;
end

expression3 = xpath.compile('systemParameters/parameter[@name="Color-Mode/Centre"]/@value');
DcNode = expression3.evaluate(file, XPathConstants.NODE);
expression33 = xpath.compile('systemParameters/parameter[@name="Power-Mode/Centre"]/@value');
Dc1Node = expression33.evaluate(file, XPathConstants.NODE);

if isempty(DcNode)
    Dc = str2double(Dc1Node.getTextContent)/skalaX;
else
    Dc = str2double(DcNode.getTextContent)/skalaX;
end

% expression4 = xpath.compile('systemParameters/parameter[@name="B-Mode/Width"]/@value');
% XbNode = expression4.evaluate(file, XPathConstants.NODE);
% Xb = str2double(XbNode.getTextContent)/skalaX;

% expression5 = xpath.compile('systemParameters/parameter[@name="B-Mode/Depth"]/@value');
% BMDNode = expression5.evaluate(file, XPathConstants.NODE);
% BMD = str2double(BMDNode.getTextContent)/skalaY;
% 
% expression6 = xpath.compile('systemParameters/parameter[@name="B-Mode/Depth-Offset"]/@value');
% BMDOffNode = expression6.evaluate(file, XPathConstants.NODE);
% BMDOff = str2double(BMDOffNode.getTextContent)/skalaY;
% 
% BMup=BMD-BMDOff;

expression7 = xpath.compile('systemParameters/parameter[@name="3D-Step-Size"]/@value');
StepSizeNode = expression7.evaluate(file, XPathConstants.NODE);
expression77 = xpath.compile('systemParameters/parameter[@name="3D-Step-Size"]/@value');
StepSize1Node = expression77.evaluate(file, XPathConstants.NODE);

if isempty(StepSizeNode)
    StepSize = str2double(StepSize1Node.getTextContent);
else
    StepSize = str2double(StepSizeNode.getTextContent);
end

X1 = parametry.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinX0;
X2 = parametry.SequenceOfUltrasoundRegions.Item_1.RegionLocationMinY0;
X3 = parametry.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxX1;
X4 = parametry.SequenceOfUltrasoundRegions.Item_1.RegionLocationMaxY1;

Wbmode=X4-X2;
Dbmode=X3-X1;

im=imcrop(stak(:,:,:,1),[X1 X2 Dbmode Wbmode]);
imgs=size(im);
R1=(imgs(2)/2)+Dc-(D/2);
R2=D;
R3=(imgs(1)/2)-(W/2);
R4=W;
I2=imcrop(im,[R1 R3 R2 R4]);
imsi = size(I2);
rozmiarx = imsi(1);
rozmiary = imsi(2);
rozmiarz = imsi(3);
rozmiars = size(stak,4);
stak2 = uint8(zeros(rozmiarx,rozmiary,rozmiarz,rozmiars));

for k=1:size(stak,4)
    im3=imcrop(stak(:,:,:,k),[X1 X2 Dbmode Wbmode]);
    I3=imcrop(im3,[R1 R3 R2 R4]);
    stak2(:,:,:,k)=I3;
end

imm = axes('Parent',f,'units','normalized',...
    'Position',[.345 .37 .33 .64]);

tabela = uitable('Parent',f,'units','normalized','Visible','off',...
    'BackgroundColor',[.79 .82 .95; .74 .82 .93],'RowStriping','on');

slice_number1 = uicontrol('Style','edit',...
    'units','normalized','String','1',...
    'Position', [.495 .44 .03 .025],...
    'Callback',@slice1_number_callback);

slider1 = uicontrol('Style','slider','Parent',f,...
    'units','normalized',...
    'Position',[.345 .41 .33 .025],...
    'String','Slice number',...
    'Callback',{@slider1_callback,imm});
set(slider1,'value',1); 
set(slider1,'max',size(stak2,4)); 
set(slider1,'min',1);
set(slider1,'Sliderstep',[1/size(stak2,4), 10/size(stak2,4)] );

guz = uicontrol('Style','pushbutton','Parent',f,...
    'units','normalized',...
    'Position',[.7 .75 .1 .04],...
    'String','Choose ROI',...
    'Callback',{@guz_callback,tabela});

 uicontrol('Style','pushbutton','Parent',f,...
    'units','normalized',...
    'Position',[.7 .68 .1 .04],...
    'String','Save image',...
    'Callback',@zapisz_callback);

uicontrol('Style','pushbutton','Parent',f,...
    'units','normalized',...
    'Position',[.7 .59 .1 .06],...
    'String','<html><p style="text-align: center;">Copy data from table<br>to clipboard</p></html>',...
    'Callback',@export_callback);

% h = uibuttongroup('units','normalized',...
%     'Position',[.2 .63 .1 .12],'Parent',f,...
%     'SelectedObject',[],'BackgroundColor',[.74 .82 .93],...
%     'shadowcolor',[.74 .82 .93]);
% 
% uicontrol('Style','radiobutton','String','Color Doppler',...
%     'units','normalized',...
%     'Position',[.017 .59 .95 .25],'parent',h);
% uicontrol('Style','radiobutton','String','Power Doppler',...
%     'units','normalized',...
%     'Position',[.017 .17 .95 .25],'parent',h);

volume = uicontrol('Style','pushbutton','Parent',f,...
    'units','normalized','Visible','off',...
    'Position',[.70 .27 .14 .04],...
    'String','Measure total vessels volume [mm3]',...
    'Callback',{@volume_callback,tabela});

volume_nc = uicontrol('Style','pushbutton','Parent',f,...
    'units','normalized','Visible','off',...
    'Position',[.55 .27 .14 .04],...
    'String','Measure red vessels volume [mm3]',...
    'Callback',{@volume_nc_callback,tabela});

volume_nb = uicontrol('Style','pushbutton','Parent',f,...
    'units','normalized','Visible','off',...
    'Position',[.85 .27 .14 .04],...
    'String','Measure blue vessels volume [mm3]',...
    'Callback',{@volume_nb_callback,tabela});

volume_guz = uicontrol('Style','pushbutton','Parent',f,...
    'units','normalized','Visible','off',...
    'Position',[.625 .17 .14 .06],...
    'String','Measure tumour volume [mm3]',...
    'Callback',{@volume_guz_callback,tabela});

volume_guz_el = uicontrol('Style','pushbutton','Parent',f,...
    'units','normalized','Visible','off',...
    'Position',[.775 .17 .14 .06],...
    'String','<html><p style="text-align: center;">Measure tumour volume<br>from ellipsoid [mm3]</p></html>',...
    'Callback',{@volume_guz_el_callback,tabela});

volume_guz_el_text = uicontrol('Style','text',...
    'units','normalized','Visible','off',...
    'Position', [.775 .145 .14 .025],...
    'BackgroundColor',[.74 .82 .93]);

volume_text = uicontrol('Style','text',...
    'units','normalized','Visible','off',...
    'Position', [.7 .245 .14 .025],...
    'BackgroundColor',[.74 .82 .93]);

volume_guz_text = uicontrol('Style','text',...
    'units','normalized','Visible','off',...
    'Position', [.625 .145 .14 .025],...
    'BackgroundColor',[.74 .82 .93]);

volume_nc_text = uicontrol('Style','text',...
    'units','normalized','Visible','off',...
    'Position', [.55 .245 .14 .025],...
    'BackgroundColor',[.74 .82 .93]);

volume_nb_text = uicontrol('Style','text',...
    'units','normalized','Visible','off',...
    'Position', [.85 .245 .14 .025],...
    'BackgroundColor',[.74 .82 .93]);

imshow(stak2(:,:,:,1),'Parent',imm);

set(findobj(gcf,'type','axes'),'hittest','off');

number_errors = 0;

stak3 = zeros(size(stak2));

    function zapisz_callback(~,~)
        klatka=getframe(imm);
        [iks,~] = frame2im(klatka);
        imshow(iks,'Parent',imm);
        imsave(imm);        
    end
    function guz_callback(~,~,tabela)
        slice = round(get(slider1,'Value'));
        im2=im2double(getimage(imm));
        immmm = impoly(imm);
        wait(immmm);
        guz = createMask(immmm);
        guz2=bsxfun(@times,im2,guz);
%         stak4(:,:,slice)=guz;
        stak3(:,:,:,slice)=guz2;
        switch isempty(W1Node)
            case 0
                set(tabela,'ColumnName',{'Slice number','Red vessels area [mm2]','Blue vessels area [mm2]','Total vessels area [mm2]','Tumour area [mm2]','Ratio'});
                set(tabela,'Position',[.071 .1 .381 .25]);
                set(tabela,'Visible','on');
                set(volume,'Visible','on');
                set(volume_text,'Visible','on');
                set(volume_guz,'Visible','on');
                set(volume_guz_text,'Visible','on');
                set(volume_nc,'Visible','on');
                set(volume_nc_text,'Visible','on');
                set(volume_nb,'Visible','on');
                set(volume_nb_text,'Visible','on');
                set(volume_guz_el_text,'Visible','on');
                set(volume_guz_el,'Visible','on');
            powierzchnia_guza=length(find(guz==1));
            im_red=stak3(:,:,1,slice);
            im_blue=stak3(:,:,3,slice);
            im_gray=rgb2gray(stak3(:,:,:,slice));
            im_diff_red=imsubtract(im_red,im_gray);
            im_diff_blue=imsubtract(im_blue,im_gray);
            TH=graythresh(im_diff_red);
            Ii=im2bw(im_diff_red,TH);
            I = bwareaopen(Ii,20);
            objectAreaInPixels_red=length(find(I==1));
            TH2=graythresh(im_diff_blue);
            Ii1=im2bw(im_diff_blue,TH2);
            I1 = bwareaopen(Ii1,20);
            objectAreaInPixels_blue=length(find(I1==1));
            powierzchnia_guza2=powierzchnia_guza*skalaX*skalaY;
            powierzchnia_naczyn_c = objectAreaInPixels_red*skalaX*skalaY;
            powierzchnia_naczyn_n = objectAreaInPixels_blue*skalaX*skalaY;
            powierzchnia_naczyn=(objectAreaInPixels_red+objectAreaInPixels_blue)*skalaX*skalaY;
            ratio=powierzchnia_naczyn/powierzchnia_guza2;
            dane={slice,powierzchnia_naczyn_c,powierzchnia_naczyn_n,powierzchnia_naczyn,powierzchnia_guza2,ratio};
            olddata = get(tabela,'Data');
            if isnan(dane{6})|| dane{6}==inf
                set(tabela,'Data',olddata);
            else
                newdata = [olddata; dane];
                set(tabela,'Data',newdata);
            end
            case 1
                set(tabela,'ColumnName',{'Slice number','Vessels area [mm2]','Tumour area [mm2]','Ratio'});
                set(tabela,'Position',[.221 .1 .231 .25]);
                set(tabela,'Visible','on');
                set(volume_nc,'Visible','off');
                set(volume_nb,'Visible','off');
                set(volume_nc_text,'Visible','off');
                set(volume_nb_text,'Visible','off');
                set(volume,'Visible','on');
                set(volume_text,'Visible','on');
                set(volume_guz,'Visible','on');
                set(volume_guz_text,'Visible','on');
                set(volume_guz_el_text,'Visible','on');
                set(volume_guz_el,'Visible','on');
            powierzchnia_guza=length(find(guz==1));
            im_red=stak3(:,:,1,slice);
            im_blue=stak3(:,:,2,slice);
            im_gray=rgb2gray(stak3(:,:,:,slice));
            im_diff_red=imsubtract(im_red,im_gray);
            im_diff_blue=imsubtract(im_blue,im_gray);
            TH=graythresh(im_diff_red);
            Ii=im2bw(im_diff_red,TH);
            I = bwareaopen(Ii,20);
            objectAreaInPixels_red=length(find(I==1));
            TH2=graythresh(im_diff_blue);
            Ii1=im2bw(im_diff_blue,TH2);
            I1 = bwareaopen(Ii1,20);
            objectAreaInPixels_blue=length(find(I1==1));
            powierzchnia_guza2=powierzchnia_guza*skalaX*skalaY;
            powierzchnia_naczyn=(objectAreaInPixels_red+objectAreaInPixels_blue)*skalaX*skalaY;
            ratio=powierzchnia_naczyn/powierzchnia_guza2;
            dane={slice,powierzchnia_naczyn,powierzchnia_guza2,ratio};
            olddata = get(tabela,'Data');
            if isnan(dane{4})|| dane{4}==inf
                set(tabela,'Data',olddata);
            else
                newdata = [olddata; dane];
                set(tabela,'Data',newdata);
            end
        end
    end
    function slider1_callback(obj,~,imm)
        slice = round(get(obj,'Value'));
        imshow(stak2(:,:,:,slice),'Parent',imm);
        set(slice_number1,'String',num2str(slice));
        drawnow();
    end

    function slice1_number_callback(obj,~)
        val = str2double(get(obj,'String'));
        if isnumeric(val) && length(val) == 1 && ...
         val >= get(slider1,'Min') && ...
         val <= get(slider1,'Max')
         set(slider1,'Value',val);
         imshow(stak2(:,:,:,val),'Parent',imm);
        else
         number_errors = number_errors+1;
         set(obj,'String',...
             ['You have entered an invalid entry ',...
             num2str(number_errors),' times.']);
        end
    end
 
    function export_callback(~,~)
        d1 = get(tabela,'Data');
        d=cell2mat(d1);
        size_d = size(d); 
        str = ''; 
        for i=1:size_d(1) 
        for j=1:size_d(2) 
        str = sprintf ( '%s%f\t', str, d(i,j) ); 
        end 
        str = sprintf ( '%s\n', str ); 
        end 
        clipboard ( 'copy', str );
    end
 
    function volume_callback(~,~,tabela)
        danee = get(tabela,'Data');
        switch isempty(W1Node)
            case 1
        danee1=cell2mat(danee(:,2));
        vol1=danee1.*StepSize;
        vol=sum(vol1(:));
        set(volume_text,'String',num2str(vol));
            case 0
        danee1=cell2mat(danee(:,4));
        vol1=danee1.*StepSize;
        vol=sum(vol1(:));
        set(volume_text,'String',num2str(vol));
        end
    end

    function volume_guz_callback(~,~,tabela)
        danee = get(tabela,'Data');
        switch isempty(W1Node)
            case 1
        danee1=cell2mat(danee(:,3));
        vol1=danee1.*StepSize;
        vol=sum(vol1(:));
        set(volume_guz_text,'String',num2str(vol));
            case 0
        danee1=cell2mat(danee(:,5));
        vol1=danee1.*StepSize;
        vol=sum(vol1(:));
        set(volume_guz_text,'String',num2str(vol));
        end
    end

    function volume_nc_callback(~,~,tabela)
        danee = get(tabela,'Data');
        danee1=cell2mat(danee(:,2));
        vol1=danee1.*StepSize;
        vol=sum(vol1(:));
        set(volume_nc_text,'String',num2str(vol));
    end

    function volume_nb_callback(~,~,tabela)
        danee = get(tabela,'Data');
        danee1=cell2mat(danee(:,3));
        vol1=danee1.*StepSize;
        vol=sum(vol1(:));
        set(volume_nb_text,'String',num2str(vol));
    end

    function volume_guz_el_callback(~,~,tabela)
        switch isempty(W1Node)
            case 0
        danee = get(tabela,'Data');
        danee1=cell2mat(danee(:,5));
        [~,B] = max(danee1);
                danee2 = cell2mat(danee(:,1));
        slice = danee2(B);
        imshow(stak3(:,:,:,slice),'Parent',imm);
        set(slider1,'Value',slice);
        set(slice_number1,'String',num2str(slice));
        hh1 = imline(imm);
        wait(hh1);
        pozycja1 = getPosition(hh1);
        hh2 = imline(imm);
        wait(hh2);
        pozycja2 = getPosition(hh2);
        a = (pozycja1(2,1)-pozycja1(1,1))*skalaX;
        b = (pozycja2(2,2)-pozycja2(1,2))*skalaY;
        c = size(danee1,1);
        vol = (6/pi)*a*b*c;
        set(volume_guz_el_text,'String',num2str(vol));
            case 1
        danee = get(tabela,'Data');
        danee1=cell2mat(danee(:,3));
        [~,B] = max(danee1);
        danee2 = cell2mat(danee(:,1));
        slice = danee2(B);
        imshow(stak3(:,:,:,slice),'Parent',imm);
        set(slider1,'Value',slice);
        set(slice_number1,'String',num2str(slice));
        hh1 = imline(imm);
        wait(hh1);
        pozycja1 = getPosition(hh1);
        hh2 = imline(imm);
        wait(hh2);
        pozycja2 = getPosition(hh2);
        a = (pozycja1(2,1)-pozycja1(1,1))*skalaX;
        b = (pozycja2(2,2)-pozycja2(1,2))*skalaY;
        c = size(danee1,1)*StepSize;
        vol = (pi/6)*a*b*c;
        set(volume_guz_el_text,'String',num2str(vol));
        end
    end
end
