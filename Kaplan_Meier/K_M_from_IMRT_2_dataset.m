function [  ] = K_M_from_IMRT_2_dataset(  )

%This fuction is intended for general use in generation of K-M survival
%plots from the pvGUI databases. All survivial information is from pvGUI
%function for the project. All other information Imaging and other data must be drawn
%from Matlab_export files in CenterDataProcessing

%for the first draft I will not support the Matlab_export system. I'll
%evenetually figure out the easy way to just input the path and get the
%most recent file.

%Matt Maggio 1/27/2016
cd('Z:\CenterMATLAB\pviewer\IMRT_Version_2')
Experiment_list = pv_database();


Boost_list = [];
Anti_boost_list = [];

Boost_surv_vec = [];
Boost_Time_till_fail_vec = [];
Anti_boost_surv_vec = [];
Anti_boost_Time_till_fail_vec = [];

Clock = clock();
Analysis_date = Clock(1:3);

    for ii = 1:length(Experiment_list);
        
        Experiment_date = Experiment_list{ii}.Experiment_date; 
       %Check if censored
       if length(Experiment_list{ii}.Censor_survivial)>3
           
       %is Boost or Antiboost    
       elseif size(Experiment_list{ii}.Boost_or_Anti_boost,2) == size('boost',2);
           Boost_list(end+1) = ii;
           Boost_surv_vec(end+1) =  Experiment_list{ii}.Survival_1_is_cure;
           if Boost_surv_vec(end) == 0 
               Boost_Time_till_fail_vec(end+1) = (Experiment_list{ii}.Failure_time_days);
           else
               Time_after_exp = Analysis_date-Experiment_date; 
               Boost_Time_till_fail_vec(end+1) = fix(Time_after_exp(1)*365 + Time_after_exp(2)*30.5 + Time_after_exp(3));
               if Boost_Time_till_fail_vec(end)>120 
                   Boost_Time_till_fail_vec(end) = 120;
               end
           end
                 
       elseif size(Experiment_list{ii}.Boost_or_Anti_boost,2) == size('anti_boost',2);
           Anti_boost_list(end+1) = ii;
           Anti_boost_surv_vec(end+1) =  Experiment_list{ii}.Survival_1_is_cure;
           if Anti_boost_surv_vec(end) == 0 
               Anti_boost_Time_till_fail_vec(end+1) = (Experiment_list{ii}.Failure_time_days);
           else
               Time_after_exp = Analysis_date-Experiment_date; 
               Anti_boost_Time_till_fail_vec(end+1) = fix(Time_after_exp(1)*365 + Time_after_exp(2)*30.5 + Time_after_exp(3));
               if Anti_boost_Time_till_fail_vec(end)>120 
                   if Anti_boost_Time_till_fail_vec()
                   Anti_boost_Time_till_fail_vec(end) = 120;
               end
           end       
       end       
       end    
    end
    
    
    
    
    Day_1_to_120_vec = [1:120];
    Boost_survival_per_cent = zeros(size(Day_1_to_120_vec));
    Anti_boost_survival_per_cent = zeros(size(Day_1_to_120_vec));
    for ii = 1:length(Day_1_to_120_vec);
    Boost_survival_per_cent(ii) = (length(Boost_list) - numel(find(Boost_surv_vec==0 & Boost_Time_till_fail_vec <= Day_1_to_120_vec(ii) )))/ length(Boost_list) ;
    Anti_boost_survival_per_cent(ii) = (length(Anti_boost_list) - numel(find(Anti_boost_surv_vec == 0 & Anti_boost_Time_till_fail_vec <= Day_1_to_120_vec(ii)))) / length(Anti_boost_list);
    end



end

