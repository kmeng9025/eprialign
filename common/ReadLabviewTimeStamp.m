% function [t1, t2] = ReadLabviewTimeStamp(strTimeStamp)
% strTimeStamp is a string of the format:
%      23:13:28:.665	23:13:30:.040

function [t1, t2] = ReadLabviewTimeStamp(strTimeStamp)

a = regexp(strTimeStamp, '(?<h>\d+):(?<m>\d+):(?<s>\d+):.(?<ms>\d+)', 'names');
t1=str2num(a(1).h)*3600+str2num(a(1).m)*60+str2num(a(1).s)+str2num(a(1).ms)*1E-3;
t2=str2num(a(2).h)*3600+str2num(a(2).m)*60+str2num(a(2).s)+str2num(a(2).ms)*1E-3;

