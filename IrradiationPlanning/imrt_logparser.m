function res = imrt_logparser(fname)

hmsv=[];
kvv=[];
mav=[];
beamon=[];
elapsed=[];
timpat='([0-9]*):([0-9]*):([0-9.]*)';
kvpat = 'kV=([0-9.]*)';
mapat = 'mA=([0-9.]*)';
rxpat='(Rx)';

fd=fopen(fname,'r');
inpoll=0;

while (~feof(fd))
    mystring=fgetl(fd);
%disp(mystring)
    if contains(mystring,'>>Poll')
        inpoll=1;
        continue;
    end
    if contains(mystring,'<<Poll')
        inpoll=0;
    end
    %inpoll
    if ~inpoll, continue, end
    rxtok=regexp(mystring,rxpat,'tokens');
    if isempty(rxtok),continue,end;
    %disp(mystring);
    kvtok=regexp(mystring,kvpat,'tokens');
    if isempty(kvtok),continue, end;
    matok=regexp(mystring,mapat,'tokens');
    if isempty(matok),continue, end;
    timtok=regexp(mystring,timpat,'tokens');
    hms=str2num(char(timtok{:}));
    kv=str2num(char(kvtok{:}));
    ma=str2num(char(matok{:}));
    ifon=(kv>0 && ma>0);
    hmsv=[hmsv;hms'];
    kvv=[kvv;kv];
    mav=[mav;ma];
    beamon=[beamon;ifon];
    deltime=(hms(1)-hmsv(1,1))*60*60+hms(2)*60+hms(3);
    elapsed=[elapsed;deltime];
    %fprintf(1,'%02d:%02d:%.03f  -  kV=%.3f, mA=%.2f\n', hms, kv, ma);
end
fclose(fd);

beamon(end+1)=0;
hmsv(end+1,:)=hmsv(end,:);
kvv(end+1)=0;
mav(end+1)=0;
elapsed(end+1)=elapsed(end);
beamtrans=diff(beamon);
ons=find(beamtrans==1);
offs=find(beamtrans==-1);
ontimes=hmsv(ons,:);
% offtimes=hmsv(offs,:);
intervals=elapsed(offs)-elapsed(ons);

FileInfo = dir(fname);
dd = datevec(FileInfo.date);

res = cell(size(ontimes,1), 1);
for ii=1:length(res)
    midpt=offs(ii)-1;
    res{ii}.Time = datetime([dd(1:3),ontimes(ii,:)]);
    res{ii}.Duration = intervals(ii);
    res{ii}.kV = kvv(midpt);
    res{ii}.mA = mav(midpt);
end