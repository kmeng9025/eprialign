%OFFICEDOC read/write/append/format/display data in Microsoft Office documents (XLS/DOC/PPT)
%
%   Syntax:
%     [file,status,errMsg] = officedoc(fileName, 'open',   propName,propValue,...)
%     [data,status,errMsg] = officedoc(file,     'read',   propName,propValue,...)
%          [status,errMsg] = officedoc(file,     'write',  propName,propValue,...)
%          [status,errMsg] = officedoc(file,     'format', propName,propValue,...)
%          [status,errMsg] = officedoc(file,     'close',  propName,propValue,...)
%                            officedoc(file,     'display', visibleFlag)
%
%                 helpText = officedoc('help', helpType)
%
%   officedoc reads/writes/appends and formats data, images & screenshots
%   in Microsoft Office documents. Supported formats include XLS (Excel),
%   DOC (Word) & PPT (PowerPoint). Opening/closing COM server connection
%   and files is user-controllable, enabling very fast sequential writes.
%   Numerous format properties enable highly-customizable output.
%
%   [FILE, STATUS, ERRMSG] = officedoc(FILENAME, 'open', ...) opens the
%   specified FILENAME for later reading/writing. The returned FILE struct
%   may be used later to read/write/format file data. The optional STATUS
%   output argument is 0 in case of sucess or -1 in case of error, when
%   the error message is specified in another optional argument ERRMSG.
%   The document format is determined from the FILENAME extension (xls,
%   doc or ppt). Use the <a href="matlab:officedoc help props">optional properties</a> to set open mode etc.
%
%   [DATA, STATUS, ERRMSG] = officedoc(FILE, 'read', ...) reads DATA from
%   specified FILE, a struct that was returned from officedoc(...,'open').
%   The basic DATA format is a structure based on the top-level division
%   of the document (worksheets, chapters or slides, depending on file
%   format). The structure fields are the division names and the field
%   contents is the data within them. If only a single division was found
%   or requested, then DATA contains the division's data immediately (not
%   contained within a structure). STATUS is -1 in case of error, or the
%   actual number of data elements/characters read in case of success.
%
%   officedoc(FILE, 'write', ...) writes data to the specified FILE.
%   Use the <a href="matlab:officedoc help props">optional properties</a> to specify range & data. If no range is
%   specified, then the last file position (before file was last closed)
%   will be used (XLS/DOC only, not PPT). STATUS is -1 in case of error,
%   or actual number of data elements/characters writen in case of success.
%   Formating properties may be specified for the written data, saving a
%   separate officedoc(FILE, 'format', ...) command.
%
%   officedoc(FILE, 'format', ...) formats data in the specified FILE.
%   Use the <a href="matlab:officedoc help props">optional properties</a> to specify range & format options.
%
%   officedoc(FILE, 'close', ...) closes the specified FILE.
%   Use the <a href="matlab:officedoc help props">optional properties</a> to specify closing properties.
%
%   officedoc(FILE, 'display', VISIBLEFLAG) displays the specified FILE in
%   a corresponding MS Office application. If FILE is closed, then FILENAME
%   should be used instead. VISIBLEFLAG is an optional flag determining
%   visibility: true/1/'on' (=default) means display; false/0/'off'=hide.
%   You can also specify this as an officedoc(...,'open',...) property.
%   Note: PPT should remain visible or some features fail (ok in DOC/XLS)
%
%   officedoc(..., propName,propValue, ...) sets the property value(s)
%   for the specified file action. Property specification order does
%   not matter. PropNames are always case-insensitive, but in a few cases
%   propValues might be case-sensitive. Some props are not supported by
%   one or more doc types: such properties are simply ignored if irrelevant.
%   66 different properties are supported (more in future versions).
%   Type "<a href="matlab:officedoc help props">officedoc help props</a>" to see the full list of supported properties.
%
%   officedoc comes a long way in enabling highly-customizable document
%   output. However, there will always be certain unsupported Office features.
%   Type "<a href="matlab:officedoc help examples">officedoc help examples</a>" to see examples of accessing and
%   using such features via direct COM calls.
%
%   officedoc('help',HELPTYPE) displays extended help on officedoc usage:
%   - officedoc('help') displays this help section (same as "help officedoc")
%   - officedoc('help','props') describes all the supported propNames
%   - officedoc('help','examples') displays officedoc usage examples
%   - officedoc('help','xls') displays only help relevant to XLS (or 'ppt','doc')
%
%   Examples: type "<a href="matlab:officedoc help examples">officedoc help examples</a>"
%
%   Known Limitations:
%     1. OPEN:   PPT must remain visible or else some features fail
%     2. READ:   Missing DivType='headings' (DOC)
%     3. READ:   Disregards specified page/line/range (DOC)
%     4. READ:   Cannot read non-text data (images, tables & other complex formats)
%     5. FORMAT: Edge properties fail in Matlab 6 (R12) due to a bug in
%                Maltab's ActiveX implementation (ok in later versions) (XLS)
%     6. FORMAT: Lots of missing properties that could be added (XLS/DOC/PPT)
%     7. OfficeDoc is designed to work on Windows machines (might work on Macs)
%
%   Bugs and suggestions:
%     OfficeDoc was tested on Office 11 (XP) & Matlab 6.0 (R12) - 7.4 (R2007a)
%     but might also work on earlier versions of Matlab & Office.
%     Please report any incompatibilities or other bugs/suggestions to:
%     YMASoftware@gmail.com
%
%   Change log:
%     2007-May-29: First version posted on <a href="http://www.mathworks.com/matlabcentral/fileexchange/loadAuthor.do?objectType=author&mfx=1&objectId=1096533#">MathWorks File Exchange</a>
%
%   See also:
%     xlsread, xlswrite, xlswrite1, docSave, pptSave (last 3 on the File Exchange - ids 340,3149,10465)

% License to use this code is granted freely without warranty to all, as long as the original author is
% referenced and attributed as such. The original author maintains the right to be solely associated with this work.

% Programmed and Copyright by Yair M. Altman (YMA Software)
% $Revision: 1.2 $  $Date: 2007/05/29 22:28:27 $

%%%
%  Supported OfficeDoc properties:
%  ==============================
%    + indicates properties supported in both demo & professional mode
%    - indicates properties supported in professional mode only
%
%  OPEN/READ/WRITE/FORMAT properties:
%    + 'Mode':    'read'/'write'/'append' (OPEN property; default=read)
%                 indicates the read/write mode of the opened file
%    + 'Display': true/false/1/0/'on','off' (OPEN property; default=false)
%    - 'Sheet':   number (index, starting at 1) or string (name) - XLS only
%    - 'Page':    number - DOC/PPT only
%    - 'Slide':   number - PPT only (equivalent to 'Page')
%    - 'Range':   XLS: string ('A1' or 'A1:B2' format)
%                 DOC/PPT: numeric array: [startChar,endChar] (e.g., [13,24])
%                          relative to the current point
%    - 'Line':    number - DOC only - relative to current position; 0=start of document
%    - 'DivType': 'Characters'/'Words'/'Sentences'/'Paragraphs'/'Content'
%                 (READ property; DOC only). Defines divisions in returned DATA
%    - 'Save':    true/false/1/0/'on','off' (default=false)
%                 If true, save the document after modification (false=much faster)
%
%  WRITE-specific properties:
%    + 'Title':   string
%    + 'Data':    general (string/numeric/matrix/cell array/...)
%    + 'Image':   string - image filename or handle of Matlab figure
%    + 'Picture': equivalent to 'Image'
%    - 'Meta':    true/false/1/0/'on','off' (default=false)
%                 Matlab figure will be pasted as meta-info, not screenshot
%                 Note: might not work if Adobe Acrobat tool is installed
%    - 'NewPage': true/false/1/0/'on','off' (default=false) - DOC only
%    - 'HeaderText': string (default = '' = empty) - DOC/XLS only
%                    XLS: single header for each worksheet separately
%                    DOC: single header for entire document
%    - 'FooterText': string (default = filename)
%                    XLS: single footer for each worksheet separately
%                    DOC: single footer for entire document
%                    PPT: separate footer for each slide separately
%
%  FORMAT-specific common properties:
%    + 'Fgcolor':       Accepts <a href="matlab:doc ColorSpec">all valid Matlab colors</a>
%    + 'Foreground':    equivalent to 'Fgcolor'
%    - 'Bgcolor':       Accepts <a href="matlab:doc ColorSpec">all valid Matlab colors</a> (''=white)
%    - 'Background':    equivalent to 'Bgcolor'
%    - 'FontSize':      number (e.g. 10)
%    - 'FontName':      string (e.g. 'Times New Roman')
%    + 'Bold':          true/false/1/0/'on','off' (default=false)
%    - 'Italic':        true/false/1/0/'on','off' (default=false)
%    - 'Subscript':     true/false/1/0/'on','off' (default=false)
%    - 'Superscript':   true/false/1/0/'on','off' (default=false)
%    - 'Shadow':        true/false/1/0/'on','off' (default=false)
%    - 'BgPicture':     string: filename (or empty '' to remove)
%    - 'PageOrientation': 'Landscape'/'Portrait'
%
%  - XLS-specific FORMAT properties:
%    - 'EdgeLeft':      number (weight:0/0.5/1/1.5/2/3/4) or cell {weight,color}
%                       set weight empty or 0 to clear all edge formatting
%    - 'EdgeRight','EdgeTop','EdgeBottom' - same type as 'EdgeLeft'
%                       All 4 'Edge' properties fail in Matlab6 (ok in ML7)
%    - 'Halign':        'right'/'center'/'left'/'justify'/'fill'/'general'
%    - 'Valign':        'top'/'center'/'bottom'/'justify' (default=bottom)
%    - 'Underline':     number: 0/1/2 (default=0)
%    - 'Strikethrough': true/false/1/0/'on','off' (default=false)
%    - 'WrapText':      true/false/1/0/'on','off' (default=false)
%    - 'NumberFormat':  string (e.g. '#,##0_);[Red](#,##0)')
%    - 'TextOrientation': degrees: -90:90 (default=0=east; 90=up,-90=down)
%    - 'RowHeight':     number
%    - 'ColWidth':      number
%    - 'RowAutoFit':    true/false/1/0/'on','off' (default=false)
%    - 'ColAutoFit':    true/false/1/0/'on','off' (default=false)
%    - 'RowHidden':     true/false/1/0/'on','off' (default=false)
%    - 'ColHidden':     true/false/1/0/'on','off' (default=false)
%    - 'SheetName':     string (some characters not allowed)
%    - 'SheetHidden':   true/false/1/0/'on','off' (default=false)
%    - 'TabColor':      Accepts <a href="matlab:doc ColorSpec">all valid Matlab colors</a> (''=white)
%
%  - DOC-specific FORMAT properties:
%    - 'LineSpacing':   number: 1/1.5/2/anything (default=1)
%                       Matlab7 also accepts strings: 'wdLineSpaceSingle' etc.
%    - 'SpaceBefore':   number (points)
%    - 'SpaceAfter':    number (points)
%    - 'Underline':     number: 0/1/2/3/4 (default=0) (3=Thick, 4=Words)
%                       Matlab7 also accepts strings: 'wdUnderlineWords' etc.
%    - 'SmallCaps':     true/false/1/0/'on','off' (default=false)
%    - 'AllCaps':       true/false/1/0/'on','off' (default=false)
%    - 'Strikethrough': true/false/1/0/'on','off' (default=false)
%    - 'Emboss':        true/false/1/0/'on','off' (default=false)
%    - 'WidowControl':  true/false/1/0/'on','off' (default=false)
%    - 'KeepWithNext':  true/false/1/0/'on','off' (default=false)
%    - 'KeepTogether':  true/false/1/0/'on','off' (default=false)
%    - 'PageBreakBefore': true/false/1/0/'on','off' (default=false)
%    - 'Bullet':        string: 'None'/'Numbered'/'Unnumbered' or 'Bullet'
%                       (Note: Unnumbered=Bullet)
%    - 'LeftIndent':    number (points)
%    - 'RightIndent':   number (points)
%    - 'Halign':        'right'/'center'/'left'/'justify'
%    - 'EdgeLeft':      number (weight:0/0.5/1/1.5/2/3/4) or cell {weight,color}
%                       set weight empty or 0 to clear all edge formatting
%    - 'EdgeRight','EdgeTop','EdgeBottom' - same type as 'EdgeLeft'
%
%  - PPT-specific FORMAT properties:
%    - 'Underline':     true/false/1/0/'on','off' (default=false)
%    - 'WrapText':      true/false/1/0/'on','off' (default=false)
%    - 'Emboss':        true/false/1/0/'on','off' (default=false)
%    - 'Bullet':        string: 'None'/'Numbered'/'Unnumbered' or 'Bullet'
%                       (Note: Unnumbered=Bullet)
%    - 'LeftIndent':    number (points)
%    - 'RightIndent':   number (points)
%    - 'LineSpacing':   number (default=1)
%    - 'SpaceBefore':   number (default=0.2)
%    - 'SpaceAfter':    number (default=0)
%    - 'IndentLevel':   number (1/2/3/4...; default=1)
%    - 'TextOrientation': number: 0 or -90 (default=0=east; -90=down)
%    - 'SlideHidden':   true/false/1/0/'on','off' (default=false)
%    - 'SlideColor':    Accepts <a href="matlab:doc ColorSpec">all valid Matlab colors</a> (''=white)
%                       Note: 'BgColor' sets text frame color, not slide color
%
%  CLOSE-specific properties:
%    + 'Release': true/false/1/0/'on','off' (default=true, i.e. release)
%                 indicates whether or not to release the COM server from
%                 memory after closing the file (recommended unless the
%                 same server will be reused shortly).
%    - 'DelStd':  true/false/1/0/'on','off' (XLS-only; default=false)
%                 deletes the standard XLS worksheets. Note that XLS files
%                 must have at least one sheet, so if the file has only the
%                 standard sheets, then the last one will not be deleted.
%
%%%

%%*
%  OfficeDoc usage examples:
%  ========================
%
%  Basic Example: (works on all formats: XLS/DOC/PPT)
%       [file,status,errMsg] = officedoc(fileName, 'open', 'mode','append');
%       status = officedoc(file, 'write', 'title','My data', 'data',[1,2,3;4,5,6], 'image',gcf, 'bold',1,'fgcolor','b');
%       status = officedoc(file, 'close');
%
%  XLS-specific Example:
%     % Open document in 'append' mode:
%       [file,status,errMsg] = officedoc('test.xls', 'open', 'mode','append');
%
%     % Write a header line to the document at a specific position:
%       status = officedoc(file, 'write', 'sheet','newSheet','Range','A2','data',{'A','B','C'});
%
%     % Format the header line (font, color, alignment, borders/edges):
%     % Note: we could also append these properties to the end of the 'write' command above
%       status = officedoc(file, 'format', 'bold','on','italic',1,'fgcolor',[1,0,0],'bgcolor','y','halign','center','EdgeBottom',{4,'b'});
%
%     % Some specific formatting not <a href="matlab:officedoc help props">currently supported</a> by officedoc:
%       set(file.fid.ActiveSheet.PageSetup, 'FirstPageNumber',5, 'CenterHeader','&"Arial,Bold"- Confidential -');
%
%     % Display the document in Excel application:
%       officedoc(file, 'display');
%
%     % Loop many times and append data to the bottom of the open document:
%       for index = 1 : 10
%         data = someComputation();  % e.g., magic(3) or {1,2,3; 'a','b','c'}
%         status = officedoc(file, 'write', 'data',data);
%       end
%
%     % Close the document, deleting standard sheets and releasing COM server:
%       status = officedoc(file, 'close', 'release',1,'delStd','on');
%
%     % Re-display document; file is no longer valid so we must use file name:
%       officedoc('test.xls', 'display');
%
%  DOC-specific Example:
%     % Open document in 'write' mode:
%       [file,status,errMsg] = officedoc('test.doc', 'open', 'mode','write');
%
%     % Write a title line at a specific position:
%       status = officedoc(file, 'write', 'title','My magic data','page',3,'line',4,'PageBreakBefore',1);
%
%     % Write data beneath the header:
%       status = officedoc(file, 'write', 'line',1,'data',magic(6),'halign','center');
%
%     % Format the data (font, color, alignment, borders/edges):
%     % Note: we could also append these properties to the end of the 'write' command above
%       status = officedoc(file, 'format', 'bold','on','italic',1,'fgcolor',[1,0,0],'bgcolor','y','EdgeBottom',{4,'b'});
%
%     % Some specific formatting not <a href="matlab:officedoc help props">currently supported</a> by officedoc:
%       set(file.fid.PageSetup.LineNumbering,'Active',1);
%
%     % Display the document in Word application:
%       officedoc(file, 'display');
%
%     % Loop many times and append data to the bottom of the open document page:
%       for index = 1 : 10
%         data = someComputation();  % e.g., magic(3) or {1,2,3; 'a','b','c'}
%         status = officedoc(file, 'write', 'data',data);
%       end
%
%     % Close the document, releasing COM server:
%       status = officedoc(file, 'close', 'release',1);
%
%     % Re-display document; file is no longer valid so we must use file name:
%       officedoc('test.doc', 'display');
%
%  PPT-specific Example:
%     % Open document in 'append' mode:
%       [file,status,errMsg] = officedoc('test.ppt', 'open', 'mode','append');
%
%     % Write a new slide with title and figure screenshot (taken via clipboard):
%       status = officedoc(file, 'write', 'title','Nice plot data','image',gcf,'meta','on');
%
%     % Format the slide footer and orientation
%     % Note: we could also append these properties to the end of the 'write' command above
%       status = officedoc(file, 'format', 'FooterText','copyright a@b.c','PageOrientation','landscape');
%
%     % Some specific formatting not <a href="matlab:officedoc help props">currently supported</a> by officedoc:
%       set(file.fid.Application.ActivePresentation.PageSetup,'FirstSlideNumber',5);
%
%     % Display the document in PowerPoint application (un-minimize and bring to front):
%       officedoc(file, 'display');
%
%     % Loop many times and append data to the bottom of slide #5:
%       for index = 1 : 10
%         data = someComputation();  % e.g., magic(3) or {1,2,3; 'a','b','c'}
%         status = officedoc(file, 'write', 'slide',5,'data',data);
%       end
%
%     % Close the document, releasing COM server:
%       status = officedoc(file, 'close', 'release',1);
%
%     % Re-display document; file is no longer valid so we must use file name:
%       officedoc('test.ppt', 'display');
%
%%*
