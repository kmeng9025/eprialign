function res = pv_get_database(dirname)

dir = pwd;
cd(dirname)
res = pv_database();
cd(dir);
