%% train_denoise_unet_trainnet_min.m
clear; clc; rng(7,'twister');

% --- paths y prefijo ---
prefix   = "Seq05VD_";
gtDir    = "/home/tonix/Documents/PhdDegreeCode/MatlabCode/PictureInOFDM/701_StillsRaw_full";
noisyDir = "/home/tonix/Documents/PhdDegreeCode/MatlabCode/PictureInOFDM/StillRawOFDM/SNR_0dB";

% --- lista .png con prefijo y cruce por nombre ---
gtFiles = dir(fullfile(gtDir, prefix + "*.png"));
names   = string({gtFiles.name});
mask    = arrayfun(@(n) isfile(fullfile(noisyDir,names(n))), 1:numel(names));
names   = names(mask);
assert(~isempty(names), "No hay pares .png con prefijo %s.", prefix);
fprintf("Pares válidos: %d\n", numel(names));

% --- split 80/20 ---
idx    = randperm(numel(names));
nTrain = floor(0.8*numel(names));
trNms  = names(idx(1:nTrain));
teNms  = names(idx(nTrain+1:end));
fprintf("Split -> Train: %d | Test: %d\n", numel(trNms), numel(teNms));

% --- carga RAM (asumido 720x960x3, uint8) ---
H=720; W=960; C=3; 
XTrain = zeros(H,W,C,numel(trNms),'single');   % noisy
YTrain = zeros(H,W,C,numel(trNms),'single');   % residual = clean - noisy
for k=1:numel(trNms)
    Ic = single(imread(fullfile(gtDir,   trNms(k))))/255;   % [0,1]
    In = single(imread(fullfile(noisyDir,trNms(k))))/255;
    XTrain(:,:,:,k) = In;
    YTrain(:,:,:,k) = Ic - In;   % objetivo: residuo
end

XTest = zeros(H,W,C,numel(teNms),'single');    % noisy
YTest = zeros(H,W,C,numel(teNms),'single');    % clean
for k=1:numel(teNms)
    Ic = single(imread(fullfile(gtDir,   teNms(k))))/255;
    In = single(imread(fullfile(noisyDir,teNms(k))))/255;
    XTest(:,:,:,k) = In;
    YTest(:,:,:,k) = Ic;
end

% --- dlarray (batch completo) ---
XTrain = dlarray(XTrain,'SSCB');
YTrain = dlarray(YTrain,'SSCB');
if canUseGPU, XTrain=gpuArray(XTrain); YTrain=gpuArray(YTrain); end

% --- red (dlnetwork, salida = residuo) ---
net = buildDenoiseUNet();  % tu constructor

% --- trainnet (sin loop) ---
opts = trainingOptions("adam", ...
    MiniBatchSize=8, ...
    MaxEpochs=30, ...
    InitialLearnRate=1e-3, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="gpu", ...
    Verbose=true);

trainedNet = trainnet(XTrain, YTrain, net, "mse", opts);

% --- test: pasada única y métricas simples ---
XTest_dl = dlarray(XTest,'SSCB'); if canUseGPU, XTest_dl=gpuArray(XTest_dl); end
Rpred = predict(trainedNet, XTest_dl);
Rpred = gather(extractdata(Rpred));
Ideno = min(max(XTest - Rpred,0),1);

% PSNR simple (sin IPT)
ps = zeros(numel(teNms),1,'single');
for k=1:numel(teNms)
    mse = mean((Ideno(:,:,:,k)-YTest(:,:,:,k)).^2,'all');
    ps(k) = 10*log10(1/max(mse,eps('single')));
end
fprintf("\nTEST PSNR medio: %.2f dB\n", mean(ps));
