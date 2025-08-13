function epr_mkdir(filename)

dirname = (filename);
if ~(exist(dirname, 'file') == 7)
  mkdir((filename))
end