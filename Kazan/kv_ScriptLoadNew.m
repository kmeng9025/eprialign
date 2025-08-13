% ListOfExtensions = kv_ScriptLoadNew('extension')
% ListOfSubTypes = kv_ScriptLoadNew('subtype', extension)
% ListOfControls = kv_ScriptLoadNew('options', file_type)
% LoadingScript = kv_ScriptLoadNew('script', file_type, file_name, options)

% All loading script should have structure
% [ax,y,dsc] = loading_script(filename, ...)
% ax - axis structure
% y - data
% dsc - unstructured/file specific information
function LoadingScript = kv_ScriptLoadNew(mode, file_type, file_name, options)

switch mode
  case 'extension'
    LoadingScript = {'*.', '*.aqs', '*.img', '*.jpg', '*.dat', '*.d00', '*.d01', ...
      '*.d02', '*.dsc', '*.exp', '*.par', '*.rmn', '*.tdms', '*.txt', '*.*'};
  case 'subtype'
    switch file_type
      case '*.*'
        LoadingScript={'ASCII', 'ASCII2D', 'XEPR', 'UofC_bin', 'RMN_1D', 'RMN_2D', 'TecMag', 'XWINNMR', 'WINNMR', 'WEBMOSS', 'OPUS_FTIR', 'image'};
      case '*.'
        LoadingScript={'ASCII', 'ASCII2D', 'RMN_1D', 'RMN_2D', 'Spinsight', 'TecMag', 'XWINNMR', 'WINNMR'};
      case '*.aqs'
        LoadingScript={'WINNMR'};
      case '*.dat'
        LoadingScript={'ASCII', 'ASCII2D', 'XEPR', 'RMNdat', 'RMNdat2D', 'transient'};
      case '*.img'
        LoadingScript={'UofC_img'};
      case '*.tdms'
        LoadingScript={'SpecMan'};
      case '*.txt'
        LoadingScript={'ASCII', 'ASCII2D'};
      case {'*.exp'}
        LoadingScript={'WIS', 'SpecMan'};
      case {'*.par', '*.dsc'}
        LoadingScript={'XEPR', 'XEPR_JSS', 'ASTASHKIN'};
      case '*.rmn'
        LoadingScript={'RMN_1D', 'RMN_2D'};
      case {'*.d00'}
        LoadingScript={'WIS'};
      case '*.d01'
        LoadingScript={'SpecMan', 'SpecManNIST'};
      case '*.d02'
        LoadingScript={'SpecManRAW'};
      case '*.jpg'
        LoadingScript={'image'};
      otherwise
        LoadingScript={'---'};
    end
  case 'script'
    switch file_type
      case {'*XWINNMR', '*.XWINNMR'}
        LoadingScript{1} = ['[ax,y,dsc]=XWINNMR(''', file_name, ''');'];
      case {'*WINNMR', '*.WINNMR', '*.aqsWINNMR'}
        LoadingScript{1} = ['[ax,y,dsc]=WINNMR(''', file_name, ''');'];
      case {'*WEBMOSS'}
        LoadingScript{1} = ['[ax,y,dsc]=moss_webbin(''', file_name, ''');'];
      case {'*.rmnRMN_1D', '*RMN_1D', '*.RMN_1D'}
        LoadingScript{1} = ['[ax,y,dsc]=rmnread(''', file_name, ''', ''type'', ''1D'');'];
      case {'*.rmnRMN_2D', '*.RMN_2D','*RMN_2D'}
        idx = get(handles.ReadOptHandle(1), 'Value');
        type = get(handles.ReadOptHandle(1), 'UserData');
        type = type{idx};
        LoadingScript{1} = ['[ax,y,dsc]=rmnread(''', file_name, ''', ''type'', ''2D'',''subtype'',''',type,''');'];
      case {'*.datRMNdat'}
        LoadingScript{1} = ['[ax,y,dsc]=rmnread(''', file_name, ''', ''type'', ''1dtxt'');'];
      case {'*.datASCII2D' '*ASCII2D', '*.txtASCII2D', '*.ASCII2D'}
        xsize = get(handles.ReadOptHandle(3), 'String');
        ysize = get(handles.ReadOptHandle(6), 'String');
        xdw = get(handles.ReadOptHandle(4), 'String');
        ydw = get(handles.ReadOptHandle(7), 'String');
        cfreq = get(handles.ReadOptHandle(9), 'String');
        LoadingScript{1} = ['[ax,y,dsc]=rmnread(''', filename, ''', ''type'', ''2dtxt'',', ...
          '''xsize'',',xsize,',''xdwell'',',xdw,',''ysize'',',ysize,',''ydwell'',',ydw,',''cfreq'',',cfreq,');'];
        LoadingScript{end+1} = ['ax.xlabel=''',get(handles.ReadOptHandle(1), 'String'),''';'];
        LoadingScript{end+1} = ['ax.ylabel=''',get(handles.ReadOptHandle(1), 'String'),''';'];
      case {'*.dscXEPR', '*.parXEPR', '*.*XEPR'}
        LoadingScript{1} = ['[ax,y,dsc]=brukerread(''', file_name, ''');'];
      case {'*.dscXEPR_JSS', '*.parXEPR_JSS'}
        LoadingScript{1} = ['[ax,y,dsc]=brukerreadJSS(''', file_name, ''');'];
      case {'*.datASTASHKIN', '*.parASTASHKIN'}
        LoadingScript{1} = ['[ax,y,dsc]=astashkinread(''', file_name, ''');'];
      case  {'*.expWIS', '*.d00WIS'}
        LoadingScript{1} = ['[ax,y,dsc]=d00read(''', file_name, ''');'];
      case  {'*.dattransient'}
        LoadingScript{1} = ['[ax,y,dsc]=ESPtransreadsdsfi(''', file_name, ''');'];
      case  {'*.expSpecMan', '*.d01SpecMan'}
        LoadingScript{1} = ['[ax,y,dsc]=kv_d01read(''', file_name, ''');'];
      case  {'*.expSpecManNIST', '*.d01SpecManNIST'}
        LoadingScript{1} = ['[ax,y,dsc]=kv_nistd01read(''', file_name, ''');'];
      case  {'*.tdmsSpecMan'}
        LoadingScript{1} = ['[ax,y,dsc]=kv_smtdmsread(''', file_name, ''');'];
      case  {'*.d02SpecManRAW'}
        LoadingScript{1} = ['[ax,y,dsc]=kv_d02read(''', file_name, ''');'];
      case  '*OPUS_FTIR'
        nn = get(handles.ReadOptHandle(1), 'Value');
        LoadingScript{1} = ['[ax,y,dsc]=opus_read(''', file_name, ''', ', num2str(nn) ,');'];
      case {'*.*ASCII', '*.datASCII', '*ASCII', '*.ASCII', '*.txtASCII'}
        delimiter = options{1};
        fstcol = options{2};
        dunit = options{3};
        LoadingScript{1} = ['[ax,y,dsc]=asciiread(''', file_name, ''',''',delimiter,''',',num2str(fstcol>2),');'];
        if fstcol==5 || fstcol==6
          LoadingScript{end+1} = 'ax.x = y(:,1); y = y(:,2:end);';
        end
        switch fstcol
          case {2,4,5,6}
            LoadingScript{end+1} = 'ysize = size(y,2);';
            LoadingScript{end+1} = 'if ~mod(ysize, 2), yy1 = reshape(y, size(y, 1), size(y,2)/2, 2); end;';
            LoadingScript{end+1} = 'if ~mod(ysize, 2), y = yy1(:,:,1)+i*yy1(:,:,2); end;';
          case 7
        end
        LoadingScript{end+1} = ['if strcmp(ax.xlabel, ''?''), ax.xlabel=''',dunit,''';end;'];
      case '*.TecMag'
        LoadingScript{1} = ['[ax,y,dsc]=TecMagread(''', file_name, ''');'];
      case '*.Spinsight'
        LoadingScript{1} = ['[ax,y,dsc]=kv_read_spinsight(''', file_name, ''');'];
      case {'*.jpgimage', '*image'}
        LoadingScript{1} = ['[y] = imread(''', file_name, ''');'];
        LoadingScript{end+1} = 'y = rgb2gray(y);';
        LoadingScript{end+1} = 'ax.type=''image'';';
        LoadingScript{end+1} = 'ax.xlabel = ''Image width, pixels'';';
        LoadingScript{end+1} = 'dsc = struct(''comment'', ''no info'');';
      case {'*UofC_bin'}
        num = get(handles.ReadOptHandle(1), 'Value');
        LoadingScript{1} = ['[ax,y,dsc] = kv_read_halpern(''', file_name, ''',''format'',',num2str(num),');'];
      case {'*.imgUofC_img'}
        num = get(handles.ReadOptHandle(1), 'Value');
        LoadingScript{1} = ['[ax,y,dsc] = kv_read_halpern(''', file_name, ''',''format'',',num2str(num),');'];
      otherwise
        LoadingScript = {};
        error(['LoadFile: Unknown format ', file_type]);
    end
  case 'options'
    LoadingScript = {};
    switch file_type
      case {'*.*ASCII', '*.datASCII', '*ASCII', '*.txtASCII', '*.ASCII'}
        LoadingScript{1}.Type = 'popupmenu';
        LoadingScript{1}.String = {'tab', 'space', ',', ';'};
        LoadingScript{1}.Data = {'\t', ' ', ',', ';'};
        LoadingScript{1}.Default = 1;
        LoadingScript{2}.Type = 'popupmenu';
        LoadingScript{2}.String = {'Y', 'rY, iY', 'X,Y', 'X,rY,iY', 'idx, X, Y', 'idx, X, rY, iY', 'X,Y1,Y2,Y3..'};
        LoadingScript{2}.Data = {1,2,3,4,5,6,7};
        LoadingScript{2}.Default = 1;
        LoadingScript{3}.Type = 'edit';
        LoadingScript{3}.String = {'Time, s'};
      case {'*.RMN_2D','*.rmnRMN_2D'}
        LoadingScript{1}.Type = 'popupmenu';
        LoadingScript{1}.String =  {'D1:Time D2:Time', 'D1:Time D2:Freq', 'D1:Freq D2:Time', 'D1:Freq D2:Freq'};
        LoadingScript{1}.Data =  {'2DTT', '2DTF', '2DFT', '2DFF'};
        LoadingScript{1}.Default = 1;
        LoadingScript{2}.Type = 'popupmenu';
        LoadingScript{2}.String = {'Not used'};
        LoadingScript{2}.Data = {1};
        LoadingScript{2}.Default = 1;
        LoadingScript{3}.Type = 'edit';
        LoadingScript{3}.String = '';
      case '*OPUS_FTIR'
        LoadingScript{1}.Type = 'popupmenu';
        LoadingScript{1}.String =   {'Interferogram', 'Baseline (fft)', 'Data-Baseline(fft)'};
        LoadingScript{1}.Data =  {1,2,3};
        LoadingScript{1}.Default = 1;
        LoadingScript{2}.Type = 'popupmenu';
        LoadingScript{2}.String = {'Not used'};
        LoadingScript{2}.Data = {1};
        LoadingScript{2}.Default = 1;
        LoadingScript{3}.Type = 'edit';
        LoadingScript{3}.String = '';
      case {'*.datASCII2D', '*ASCII2D', '*.txtASCII2D', '*.ASCII2D'}
%         if isempty(handles.ReadOptHandle) % create
%           % control for unit
%           handles.ReadOptHandle(1) = uicontrol(handles.MainFigure, 'Style', 'edit',...
%             'String', 'Time, s', 'Units', 'normalized', ...
%             'Position', [controlLeft,controlTop(4),sz,controlHeight], ...
%             'Callback', 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))');
%           % control for X: label, xsize, dwelltime
%           handles.ReadOptHandle(2) = uicontrol(handles.MainFigure, 'Style', 'text',...
%             'String', 'X: n,dw', 'Units', 'normalized', ...
%             'Position', [controlLeft,controlTop(3),controlWidth3,controlHeight],...
%             'Callback', 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))');
%           handles.ReadOptHandle(3) = uicontrol(handles.MainFigure, 'Style', 'edit',...
%             'String', '1', 'Units', 'normalized', ...
%             'Position', [controlLeft32,controlTop(3),controlWidth3,controlHeight],...
%             'Callback', 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))');
%           handles.ReadOptHandle(4) = uicontrol(handles.MainFigure, 'Style', 'edit',...
%             'String', '1', 'Units', 'normalized', ...
%             'Position', [controlLeft33,controlTop(3),controlWidth3,controlHeight],...
%             'Callback', 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))');
%           % control for Y: label, xsize, dwelltime
%           handles.ReadOptHandle(5) = uicontrol(handles.MainFigure, 'Style', 'text',...
%             'String', 'Y: n,dw', 'Units', 'normalized', ...
%             'Position', [controlLeft,controlTop(2),controlWidth3,controlHeight],...
%             'Callback', 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))');
%           handles.ReadOptHandle(6) = uicontrol(handles.MainFigure, 'Style', 'edit',...
%             'String', '1', 'Units', 'normalized', ...
%             'Position', [controlLeft32,controlTop(2),controlWidth3,controlHeight],...
%             'Callback', 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))');
%           handles.ReadOptHandle(7) = uicontrol(handles.MainFigure, 'Style', 'edit',...
%             'String', '1', 'Units', 'normalized', ...
%             'Position', [controlLeft33,controlTop(2),controlWidth3,controlHeight],...
%             'Callback', 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))');
%           % control for central frequency
%           handles.ReadOptHandle(8) = uicontrol(handles.MainFigure, 'Style', 'text',...
%             'String', 'C. Freq', 'Units', 'normalized', ...
%             'Position', [controlLeft,controlTop(1),controlWidth2,controlHeight],...
%             'Callback', 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))');
%           handles.ReadOptHandle(9) = uicontrol(handles.MainFigure, 'Style', 'edit',...
%             'String', '0', 'Units', 'normalized', ...
%             'Position', [controlLeft2,controlTop(1),controlWidth2,controlHeight],...
%             'Callback', 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))');
%           guidata(handles.MainFigure, handles);
%         end
      case {'*UofC_bin', '*.imgUofC_img'}
        LoadingScript{1}.Type = 'popupmenu';
        LoadingScript{1}.String = {'Modula', 'LabView'};
        LoadingScript{1}.Data = {1,2};
        LoadingScript{1}.Default = 1;
        LoadingScript{2}.Type = 'popupmenu';
        LoadingScript{2}.String = {'no options'};
        LoadingScript{2}.Data = {1};
        LoadingScript{2}.Default = 1;
        LoadingScript{3}.Type = 'edit';
        LoadingScript{3}.String = {''};
    end
end
