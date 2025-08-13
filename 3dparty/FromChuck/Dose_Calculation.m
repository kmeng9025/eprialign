function [DRate,Beam_Time]=Dose_Calculation(Beam_Dose,Beam_Depth,Col_radius,BeamIndex,DoseInfo,TissueFactor)
 
    d=Beam_Depth;
    FS=Col_radius(1);

% % % % % % % %     Col_Type=DoseInfo{i,1}
% % % % % % % %     kV=DoseInfo{i,2}
% % % % % % % %     mA=DoseInfo{i,3}
% % % % % % % %     Filter=DoseInfo{i,4}
% % % % % % % %     Depths=DoseInfo{i,5}
% % % % % % % %     DoseRates=DoseInfo{i,6}
% % % % % % % %     FieldSizes=DoseInfo{i,7}
% % % % % % % %     Col_Description=DoseInfo{i,8}
% % % % % % % %     EndEff=DoseInfo{i,9}
    
    %%% Populate collimator information %%%
    Col_Type=DoseInfo{BeamIndex,1}(1);
    Col_Type_numberof=DoseInfo{BeamIndex,1}(2);
    Depths=DoseInfo{BeamIndex,5};
    DoseRates=DoseInfo{BeamIndex,6};
    FieldSizes=DoseInfo{BeamIndex,7};
    EndEff=DoseInfo{BeamIndex,9};
    
    %%% Verify d lies within measured depths
    if d>max(Depths)
        d=max(Depths);
    elseif d<min(Depths)
        d=min(Depths);
    end
    
    %%% Find which two measured depths d lies between - for interpolation later on %%%
    for i=1:length(Depths)-1
        if d>=Depths(i) && d<=Depths(i+1)
            Depths_index1=i;
            Depths_index2=i+1;
        end
    end
    d1=Depths(Depths_index1);
    d2=Depths(Depths_index2);
%     Depths_1cmindex=find(Depths==1);
    
    %%% Dose rate calculation %%%
    if strcmp(Col_Type,'Circular') && strcmp(Col_Type_numberof,'Multiple')
        %%% If field size is circular, and Collimator type is multiple (i.e. can interpolate between field sizes), find
        %%% which two field measured field sizes the desired FS lies between
        
        if FS>max(FieldSizes) % Verify desired FS isn't larger than max. measured field size.
            FS=max(FieldSizes);
        elseif FS<min(FieldSizes) % Verify desired FS isn't smaller than min. measured field size.
            FS=min(FieldSizes);
        end
        
        for i=1:length(FieldSizes)-1
            if FS<=FieldSizes(i) && FS>=FieldSizes(i+1)
                FS_index1=i+1;
                FS_index2=i;
            end
        end
        FS1=FieldSizes(FS_index1);
        FS2=FieldSizes(FS_index2);
        
        DRate11=DoseRates(Depths_index1,FS_index1);
        DRate12=DoseRates(Depths_index1,FS_index2);
        DRate21=DoseRates(Depths_index2,FS_index1);
        DRate22=DoseRates(Depths_index2,FS_index2);
        
        %%% Bi-linear interpolation to find dose rate at given depth/field size %%%
        AA=1/((d2-d1)*(FS2-FS1));
        DRate=AA*(DRate11*(d2-d)*(FS2-FS) + DRate21*(d-d1)*(FS2-FS) + DRate12*(d2-d)*(FS-FS1) + DRate22*(d-d1)*(FS-FS1));
%         PDD_1FS=PDDsF31 + (FS-FS1)*(PDDsF32-PDDsF31)/(FS2-FS1);
    elseif strcmp(Col_Type,'Circular') && strcmp(Col_Type_numberof,'Single') %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        DRate11=DoseRates(Depths_index1);
        DRate12=DoseRates(Depths_index2);
%         PDDsF31=PDDsF(Depths_1cmindex)
        
        %%% Linear interpolation to find depth at given depth, for given field size %%%
        DRate=DRate11 + (d-d1)*(DRate12-DRate11)/(d2-d1);
%         PDD_1FS=PDDsF31;
    elseif strcmp(Col_Type,'Rectangular');
        DRate11=DoseRates(Depths_index1);
        DRate12=DoseRates(Depths_index2);
%         PDDsF31=PDDsF(Depths_1cmindex)
        
        %%% Linear interpolation to find depth at given depth, for given field size %%%
        DRate=DRate11 + (d-d1)*(DRate12-DRate11)/(d2-d1);
%         PDD_1FS=PDDsF31;
    else % If not Circular or Rectangular, assume information is from one field size, i.e. same format as Rectangular
        DRate11=DoseRates(Depths_index1);
        DRate12=DoseRates(Depths_index2);
%         PDDsF31=PDDsF(Depths_1cmindex);
        
        %%% Linear interpolation to find depth at given depth, for given field size %%%
        DRate=DRate11 + (d-d1)*(DRate12-DRate11)/(d2-d1);
%         PDD_1FS=PDDsF31;
    end
    
    %%% Modify dose rate to include tissue factor multiplier %%%
    DRate=DRate*TissueFactor;
    
    %%% Test output (comment out when not necessary) %%%
% % % %     EndEff;
% % % %     Beam_Dose;
    
    %%% Find beam on time in order to achieve desired dose at desired depth, given calculated dose rate.
    %%% Note that this includes factoring in the measured end-effect of the x-ray source.
    Beam_Time=Beam_Dose/(DRate/60)+EndEff;
    
    
    