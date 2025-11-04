folder = '../../Data/kaggle_dataset/';  % You specify this!
currentPath = pwd;
disp(currentPath);
addpath(folder)  

fullMatFileName = fullfile(folder,  'v2v80211p_LOS.mat');
if ~exist(fullMatFileName, 'file')
  message = sprintf('%s does not exist', fullMatFileName);
  uiwait(warndlg(message));
else
  LOS = load(fullMatFileName);
  LOS = LOS.data;
end


fullMatFileName = fullfile(folder,  'v2v80211p_NLOS.mat');
if ~exist(fullMatFileName, 'file')
  message = sprintf('%s does not exist', fullMatFileName);
  uiwait(warndlg(message));
else
  NLOS = load(fullMatFileName);
  NLOS = NLOS.data;
end
