I   = imread("noise_image.jpg");
[noisyR,noisyG,noisyB] = imsplit(I);
noisyR = single(noisyR);
noisyG = single(noisyG);
noisyB = single(noisyB);
%net = denoisingNetwork("dncnn");
% Importa la red (tipo dlnetwork, sin inicializar)
net = importNetworkFromONNX("unet_model.onnx");
findPlaceholderLayers(net)

% PASO 1: Crear un dummy input con el mismo tamaño espacial que cada canal
[height, width] = size(noisyR); 
dummyInput = dlarray(zeros(height, width, 1, 1, 'single'), 'SSCB');
    % 'SSCB' = (Spatial, Spatial, Channel, Batch)
    % Usamos 1 canal y batch de 1

% PASO 2: Inicializar la red
net = initialize(net, dummyInput);

% PASO 3: Convertir cada canal a dlarray y predecir

% --- Canal R ---
noisyR_dl = dlarray(noisyR, 'SSCB');  % Agregamos etiquetas de dimensión
denoisedR_dl = predict(net, noisyR_dl);
denoisedR = gather(extractdata(denoisedR_dl));  % Convertimos de dlarray a array normal

% --- Canal G ---
noisyG_dl = dlarray(noisyG, 'SSCB');
denoisedG_dl = predict(net, noisyG_dl);
denoisedG = gather(extractdata(denoisedG_dl));

% --- Canal B ---
noisyB_dl = dlarray(noisyB, 'SSCB');
denoisedB_dl = predict(net, noisyB_dl);
denoisedB = gather(extractdata(denoisedB_dl));

% Reconstruir la imagen RGB
denoisedRGB = cat(3, denoisedR, denoisedG, denoisedB);
imwrite(denoisedRGB, 'clean_image.jpg');

% Comparar PSNR con una imagen limpia "original_image.jpg"
Clean = imread("original_image.jpg");
noisyPSNR = psnr(I, Clean);
fprintf("\n The PSNR value of the noisy image is %0.4f.\n", noisyPSNR);