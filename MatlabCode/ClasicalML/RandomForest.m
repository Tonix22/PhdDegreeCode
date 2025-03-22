close all;
clc;
addpath('../Libraries');

%% 1. Crear carpeta de resultados si no existe
resultsFolder = 'RandomForestResults';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

%% 2. Parámetros
SNR_dB_value = 15;    % SNR en dB
M = 4;                % Orden de modulación (QPSK)
FFTSize = 48;         % Tamaño de la FFT para OFDM
numSC = 48;           % Número de subportadoras
numFrames = 1000;     % Cantidad de tramas a simular
H = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;

%% 3. Generar datos y acumular en X (características) e Y (etiquetas)
allFeatures = [];  % Se guardarán [parteReal, parteImaginaria]
allLabels   = [];  % Se guardarán las etiquetas (0, 1, 2, 3)

for j = 1:numFrames
    % Generar símbolos aleatorios para una trama OFDM de 48 subportadoras
    signalTx = generateRandomData(M, numSC);
    
    % Transmitir por el canal y obtener la señal estimada (DPSKSignalEstimate) y sus etiquetas
    [Labels, DPSKSignalEstimate] = processChannelAndTransmit(signalTx, M, FFTSize, SNR_dB_value, numSC, H);
    
    % Partes real e imaginaria de las 48 subportadoras
    realPart = real(DPSKSignalEstimate(:));
    imagPart = imag(DPSKSignalEstimate(:));
    
    % Acumular en las matrices finales
    allFeatures = [allFeatures; realPart, imagPart];
    allLabels   = [allLabels; Labels(:)];
end

% Asignar a X e Y para mayor claridad
X = allFeatures; 
Y = allLabels;

%% 4. Mezclar aleatoriamente (shuffle) y dividir en entrenamiento y prueba
rng(0);  % Semilla para reproducibilidad
perm = randperm(size(X,1));
X = X(perm,:);
Y = Y(perm);

trainRatio = 0.8;  % 80% para entrenamiento, 20% para prueba
numTrain = round(trainRatio * size(X,1));

Xtrain = X(1:numTrain,:);
Ytrain = Y(1:numTrain,:);
Xtest  = X(numTrain+1:end,:);
Ytest  = Y(numTrain+1:end,:);

% Convertir las etiquetas a categórico para TreeBagger
YtrainCat = categorical(Ytrain);
YtestCat  = categorical(Ytest);

%% 5. Entrenar el modelo Random Forest
numTrees = 100;  % Número de árboles en el bosque aleatorio
RFModel = TreeBagger(numTrees, Xtrain, YtrainCat, ...
    'OOBPrediction','on', ...
    'Method','classification');

%% 6. Predecir sobre el conjunto de prueba
predLabels = predict(RFModel, Xtest);
% Convertir las etiquetas predichas a categóricas para que coincidan con YtestCat
predLabelsCat = categorical(predLabels);

%% 7. Graficar la matriz de confusión (sin mostrar ventana)
fig = figure('visible','off');
cm = confusionchart(YtestCat, predLabelsCat, ...
    'Title','Matriz de Confusión - Random Forest (QPSK)', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

%% 8. Guardar la figura de la matriz de confusión
saveas(fig, fullfile(resultsFolder, 'ConfusionMatrix.png'));
