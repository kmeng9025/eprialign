function [Col_Type,kV,mA,Filter,Depths,DoseRates,FieldSizes,Col_Description,EndEff,Comment]=CollimatorDoseRates(DefaultPathname,filename)

    % clc

    %%% Open file, import data from file %%%
    fid=fopen([DefaultPathname,'/',filename]);
    Comment='';
    %%% Get the first line %%%
    tline=FileLineRead(fid); % FileLienRead is a function (at the end of this file) to read in the lines of the text file.
    Col_Type_tmp1=tline; % First line should be collimator type - Circular/Rectangular
    
    %%% Get the next line - it will say if the information in the file is for a "Single" collimator
    %%% or "Multiple" collimators %%%
    tline=FileLineRead(fid);
    Col_Type_tmp2=tline;
    Col_Type={Col_Type_tmp1,Col_Type_tmp2}; % Set collimator info 
    % % % % % Col_Type{1}, Col_Type{2}
    
    %%% Get the next line, which will be kV setting for this/these collimators %%%
    tline=FileLineRead(fid);
    [T,R]=strtok(tline,'='); 
    kV=str2num(R(2:end)); % Set kV
    
    %%% Get the next line, which will be the mA setting for this/these collimators %%%
    tline=FileLineRead(fid);
    [T,R]=strtok(tline,'=');
    mA=str2num(R(2:end))';  % Set mA
    
    %%% Get the next line, which should be filter information for this/these collimators
    tline=FileLineRead(fid);
    [T,R]=strtok(tline,'=');
    Filter=R(2:end); % Set filter text information
    
%     tline=FileLineRead(fid);   %%% Get next line - don't need to save (should just be word "Depth") %%%

    %%% Get the next line - will be vector of depths at which dosimetry was done %%%
    tline=FileLineRead(fid);
    Depths=sscanf(tline,'%f')'; % Set depths
    
%     tline=FileLineRead(fid);  %%% Get next line - will be "DoseRates" text (no need to save) %%%

    %%% Need to save doserates next %%%
    DoseRates=[];
    if strcmp(Col_Type(1),'Circular') && strcmp(Col_Type(2),'Multiple')
        tline=FileLineRead(fid);   % Get next line - should be collimator field sizes if "Circular"
        FieldSizes=sscanf(tline,'%f')';
        Col_Description=['Circular, ',num2str(kV),'kV, ',num2str(mA),'mA'];
        tline=FileLineRead(fid);
        [T,R]=strtok(tline,'=');
%         while strcmp(tline,'End Effect')==0  %%% Loop until "End Effect" text found (will denote end of Dose rate values %%%
        while isempty(R)
            DoseRates=[DoseRates;sscanf(tline,'%f')'];  %%% Populate PDDsF vector/matrix %%%
            tline=FileLineRead(fid);
            [T,R]=strtok(tline,'=');
        end
    elseif strcmp(Col_Type(1),'Circular') && strcmp(Col_Type(2),'Single')
        tline=FileLineRead(fid);
        FieldSizes=sscanf(tline,'%f')';
        Col_Description=[num2str(FieldSizes(1)),'cm radius, ', num2str(kV),'kV, ',num2str(mA),'mA'];
        tline=FileLineRead(fid);
        [T,R]=strtok(tline,'=');
%         while strcmp(tline,'End Effect')==0
        while isempty(R)
            DoseRates=[DoseRates;sscanf(tline,'%f')'];
            tline=FileLineRead(fid);
            [T,R]=strtok(tline,'=');
        end
    elseif strcmp(Col_Type(1),'Rectangular') % Rectangular currently only single collimators - no interpolation between Field Sizes.
        tline=FileLineRead(fid);
    %     Col_Description=[tline,', ',num2str(kV),'kV, ',num2str(mA),'mA'];
    %     FieldSizes=0;
        FieldSizes=sscanf(tline,'%f')';
        Col_Description=[num2str(FieldSizes(1)),'cm X ',num2str(FieldSizes(2)),'cm, ',num2str(kV),'kV, ',num2str(mA),'mA'];
        tline=FileLineRead(fid);
        [T,R]=strtok(tline,'=');
%         while strcmp(tline,'End Effect')==0
        while isempty(R)
            DoseRates=[DoseRates;sscanf(T,'%f')'];
            tline=FileLineRead(fid);
            [T,R]=strtok(tline,'=');
        end
    end
%     tline=FileLineRead(fid);  
    
    %%% Get next line - should be end effect value %%%
    EndEff=str2num(R(2:end));
    Comment=Col_Description;
    tline=FileLineRead(fid);
    if length(tline) > 1, 
        Comment=tline;
    end
    fclose(fid);  %%% Close file %%%
    
    % % %         Col_Type=DoseInfo{i,1}
    % % %         kV=DoseInfo{i,2}
    % % %         mA=DoseInfo{i,3}
    % % %         Filter=DoseInfo{i,4}
    % % %         Depths=DoseInfo{i,5}
    % % %         DoseRates=DoseInfo{i,6}
    % % %         FieldSizes=DoseInfo{i,7}
    % % %         Col_Description=DoseInfo{i,8}
    % % %         EndEff=DoseInfo{i,9}
    
function linedata=FileLineRead(fid)
    % Function to read in a line of the file fid.  This function is designed
    % to read in a line from the file fid (already opened).  It is also
    % designed to skip over commented lines (commented lines are lines that start
    % with "#".
    
    closeflag=0;
    lineread=fgetl(fid);
    while closeflag==0
        if strcmp(lineread(1),'#')
            lineread=fgetl(fid);
        else
            closeflag=1;
        end
    end
    linedata=lineread;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%