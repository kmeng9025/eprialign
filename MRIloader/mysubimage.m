function mysubimage (img,r,c,gaps,sgaps,mylim,mycmap,mytitle)
% This function displays images img(:,:,i) tightly in the figure;
% row by row statrting from the frist img(:,;,1)
% r and c are the number of rows and columns respectively
% gaps are the gaps between columns and rows respectively.
% Default: gaps [] = [0.0025 0.0025]
% mylim is the scale for imagesc, mycmap is the color map you want to  use
% the default scale for imagesc is 10% of max to 90% of max and
% the default colormap is hot.
% Written by Omer Demirkaya	Dec 17 2002 
% copied from Matlab newsgroup
% http://groups.google.com/group/comp.soft-sys.matlab/browse_thread/thread/
% 26eb68333a5c4134/bceed2b6c1979c85?lnk=gst&q=axes+never+use+subplot&rnum=6&hl=en#bceed2b6c1979c85
% Modified by C. Haney based on post from Derek Goring Dec 17 2002
% in the Matlab newsgroup
% mysubimage (img,r,c,gaps,sgaps,mylim,mycmap,mytitle);

myfigOld = findall(0,'tag','MySubImage');
%myfigOlder = findall(0,'tag','MySubImageOld');
if isempty(myfigOld)
    myfigMon=figure(10);    
else
    %myfigOld = evalin('base',myfigMon);
    myfigMon=figure;
    %set(myfigOld,'tag','MySubImageOld');    
end
set(myfigMon,'tag','MySubImage');
set(myfigMon,'Name','Image Montage', 'NumberTitle','off');
% set(myfigMon,'PaperSize', [11 17])
% orient(myfigMon,'Landscape')

if (nargin < 8), mytitle=cell(1,r*c); end
if (nargin < 7)
    error('Not enough input arguments... USAGE:mysubimage(img,r,c,[0.001 0.001])');
end

if ~iscell(mytitle) && nargin == 8
    disp('Titles should be in cell structure')
    mytitle=cell(1,r*c); % resetting to empty
end
Nslices = size(img,3);

if isempty(gaps)
    if nargin == 8
        gaps = [0.075 0.075];
    else
        gaps = [];
    end
end
if isempty(sgaps)
    if nargin == 8
        sgaps = [0.16 0.009];
    else
        sgaps = [];
    end
end
if isempty(mycmap)
    mycmap='hot';
end
if isempty(mylim)
    [N, myHist] = hist(img(:),100);
%     mylim=[max(img(:))*0.1 max(img(:))*0.9];
    ins=cumtrapz(N(:))/sum(N(:));
    mylim = [myHist(max(find(ins <= 0.01))) myHist(max(find(ins <= 0.985)))];
end
pst=CalcAxesPos(r,c,gaps, sgaps);

for thisslice=1:Nslices
    axes('Position',pst(thisslice,:))
    imagesc(img(:,:,thisslice),[mylim(1) mylim(2)]);
    if ~isempty(mytitle{thisslice}), title(mytitle{thisslice},'Color','k','FontName','Arialnarrow','FontSize',8), end
    % tumor displayed correctly if data is new i.e., permute x-y
    set(gca,'YDir','Normal') % PV 5.0 display OK w/ default Matlab, CH 02-07-10
%     set(gca,'YDir','Reverse') % old code with HeadFirst/Prone needs this
    % set(gca,'YDir','Normal') is the default in Matlab, Y lowest to
    % highest
    axis image;   axis off
    colormap(mycmap);
end