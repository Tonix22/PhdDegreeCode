close all;
clear all;
clc;
addpath('../Libraries');

channelAWGN = 1; % Selecciona si es awgn o canal

%% 1. Crear carpeta de resultados "LSTMResults" si no existe
if channelAWGN
    resultsFolder = 'LSTMResultsAwgn';
else
    resultsFolder = 'LSTMResultsV2VChannel';
end

if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

%% 2. Parámetros comunes del sistema
EbNo = 0:2:12;           % Rango de EbNo

FFTSize = 48;            % Tamaño de la FFT para OFDM
M = 4;                   % Orden de modulación (QPSK)
k = log2(M);             % Bits por símbolo (para QPSK: 2)
numSC = FFTSize;              % Número de subportadoras
numBitSymbol = numSC * k; % Bits totales por símbolo OFDM
numFramesTrain = 5000;   % Número de tramas para generar datos de entrenamiento

if channelAWGN
    snrValues = EbNo + 10*log10(k);
else
    snrValues = EbNo + 10*log10(numBitSymbol); %#ok<UNRCH>
    H = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;
end


%% 3. Preallocar vectores para Test
XTestCell = cell(length(snrValues),numFramesTrain);
YTestCell = cell(length(snrValues),numFramesTrain);
SNRTestCell = cell(length(snrValues),numFramesTrain);

%% 4. Bucle principal para almacenar datos de prueba
for i = 1:length(snrValues)
    currentSNR = snrValues(i);
    fprintf('Procesando SNR = %d dB\n', currentSNR);
    for j = 1:numFramesTrain
         % Generar bits fijos alternando entre 0 y 1
         signalTxBits = randi([0 1], numBitSymbol,1);
         signalTx = bit2int(signalTxBits, k);               % Convertir bits a símbolos
         
         % Transmitir por el canal al SNR actual
         if channelAWGN
             [Labels, DPSKSignalEstimate] = processChannelAndTransmit(signalTx, M, FFTSize, currentSNR, numSC);
         else
             % Uses Channel
             [Labels, DPSKSignalEstimate] = processChannelAndTransmit(signalTx, M, FFTSize, currentSNR, numSC, H); %#ok<UNRCH>
         end
        % Convertir DPSKSignalEstimate a secuencia: matriz de 2xnumSC (2 features: real e imaginario)
        featuresSeq = [ real(DPSKSignalEstimate(:)).' ; imag(DPSKSignalEstimate(:)).' ];  % [2 x numSC]
        
        %Convertir signalTx a etiquetas categóricas (ground truth)
        labelsSeq = categorical(signalTx(:).', 0:M-1);  % fila de 1 x numSC con clases {0, 1, 2, 3}
        SNRTestCell{i,j} = currentSNR;
        XTestCell{i,j} = featuresSeq;
        YTestCell{i,j} = labelsSeq;
    end
end
%% 6. Mezclar los datos de prueba

% Convertir las celdas en arreglos lineales
XTestLinear = reshape(XTestCell, [], 1);
YTestLinear = reshape(YTestCell, [], 1);
SNRTestLinear = reshape(SNRTestCell, [], 1);

% Generar un orden aleatorio de índices
numSamples = length(XTestLinear);
randomIndices = randperm(numSamples);

% Reorganizar los datos según los índices aleatorios
XTestShuffled = XTestLinear(randomIndices);
YTestShuffled = YTestLinear(randomIndices);
SNRTestShuffled = SNRTestLinear(randomIndices);

fprintf('Los datos de prueba han sido mezclados correctamente.\n');


%% 6. Configuración y búsqueda de archivos de red
resultsFolder = 'LSTMResultsAwgn'; % Ajusta la carpeta según corresponda
matFiles = dir(fullfile(resultsFolder, 'LSTM_Network_SNR_*dB.mat'));
numNetworks = length(matFiles);
fprintf('Se encontraron %d redes en la carpeta %s\n', numNetworks, resultsFolder);

% Extraer los valores de SNR de los nombres de los archivos
snrValuesFromFiles = zeros(numNetworks, 1);
for i = 1:numNetworks
    % Ajustar la expresión regular para manejar notación científica
    snrMatch = regexp(matFiles(i).name, 'LSTM_Network_SNR_([\d\.eE\+\-]+)dB', 'tokens');
    if ~isempty(snrMatch)
        snrValuesFromFiles(i) = str2double(snrMatch{1}{1}); % Convertir a número
    else
        error('No se pudo extraer el valor de SNR del archivo: %s', matFiles(i).name);
    end
end

% Ordenar los archivos por los valores de SNR
[~, sortedIndices] = sort(snrValuesFromFiles); % Obtener los índices de orden
matFiles = matFiles(sortedIndices); % Reordenar los archivos según SNR

% Crear un arreglo de celdas para almacenar los objetos cargados
networkObjects = cell(1, numNetworks);

% Cargar cada archivo .mat y almacenar el contenido en el arreglo
for i = 1:numNetworks
    matFilePath = fullfile(resultsFolder, matFiles(i).name);
    fprintf('Cargando archivo: %s\n', matFiles(i).name);
    data = load(matFilePath); % Cargar el archivo .mat
    networkObjects{i} = data; % Almacenar el contenido en el arreglo
end

% Opcional: Mostrar un mensaje indicando que la carga ha finalizado
fprintf('Se han cargado todos los archivos .mat de la carpeta %s.\n', resultsFolder);

%% 7. Evaluación de cada red 
YPred = zeros(numNetworks, numSC);

numError_NN = zeros(length(snrValues),1);
numBits = zeros(length(snrValues),1);

for i = 1:length(XTestShuffled)
    currentSNR = SNRTestShuffled{i};
    currentX   = XTestShuffled{i};
    currentY   = YTestShuffled{i};
    for n = 1:numNetworks
        net = networkObjects{n}.net;
        YPredProbs = predict(net, {currentX});
        % Obtener la probabilidad máxima y la clase correspondiente
        [~, maxClass] = max(YPredProbs{1}, [], 1); % Máxima probabilidad y su índice (clase)
        YPred(n,:) = maxClass-1;  % Etiqueta para la secuencia
    end
    maxVotevalue = categorical(mode(YPred, 1)); % Votación
    
    % Contar errores
    numErrorCalculate_NN = sum(currentY ~= maxVotevalue);
    % Acumular errores y bits procesados del SNR actual
    snrIndex = find(snrValues == currentSNR, 1); % Encuentra el índice correspondiente

    % Usar el índice para actualizar numError_NN y numBits
    numError_NN(snrIndex) = numError_NN(snrIndex) + numErrorCalculate_NN;
    numBits(snrIndex) = numBits(snrIndex) + numBitSymbol;

    % Actualizar barra de progreso cada 100 datos
    if mod(i, 100) == 0 || i == length(XTestShuffled)
        percentComplete = (i / length(XTestShuffled)) * 100;
        fprintf('Progreso: %.2f%% (%d de %d)\n', percentComplete, i, length(XTestShuffled));
    end

end

%% 8. Evaluación de resultados

BER_NN_vals = numError_NN ./ numBits; % Calcular BER para cada SNR

% Verificar si hay divisiones por cero
if any(numBits == 0)
    error('Algunos valores de numBits son cero, no se puede calcular el BER.');
end

% Guardar los datos en un archivo .csv
results = [snrValues(:), BER_NN_vals(:)];
csvFileName = fullfile(resultsFolder, 'BER_vs_SNR_vote.csv');
writematrix(results, csvFileName);
fprintf('Datos de BER guardados en: %s\n', csvFileName);

% Crear la gráfica con semilogy
figure;
semilogy(snrValues, BER_NN_vals, '-o', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('SNR (dB)');
ylabel('BER');
title('BER vs SNR');
legend('BER (NN)', 'Location', 'southwest');

% Guardar la gráfica como un archivo .png
pngFileName = fullfile(resultsFolder, 'BER_vs_SNR.png');
saveas(gcf, pngFileName);
fprintf('Gráfica de BER guardada en: %s\n', pngFileName);