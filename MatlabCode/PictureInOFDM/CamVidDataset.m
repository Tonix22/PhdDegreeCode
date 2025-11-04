downloadFolder = tempdir;
url = "http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData"
urlImages = url + "/files/701_StillsRaw_full.zip";
urlLabels = url + "/data/LabeledApproved_full.zip";

dataFolder = fullfile(downloadFolder,"CamVid");
dataFolderImages = fullfile(dataFolder,"images");
dataFolderLabels = fullfile(dataFolder,"labels");

filenameLabels = fullfile(dataFolder,"labels.zip");
filenameImages = fullfile(dataFolder,"images.zip");

if ~exist(filenameLabels,"file") || ~exist(imagesZip,"file")   
    mkdir(dataFolder)
    
    disp("Downloading CamVid data set images (557 MB)... ");
    websave(filenameImages, urlImages);       
    unzip(filenameImages,dataFolderImages);
    disp("Done.")
   
    disp("Downloading CamVid data set labels (16 MB)... ");
    websave(filenameLabels,urlLabels);
    unzip(filenameLabels,dataFolderLabels);
    disp("Done.")
end