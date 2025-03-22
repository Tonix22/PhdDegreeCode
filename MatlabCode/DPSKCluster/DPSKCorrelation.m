close all;
clc;
addpath('../Libraries');
addpath('BinaryClassifier');

%% Parameter System
SNR_dB = 35:5:35;         % Range of SNR values in dB
M = 4;                   % Modulation order (QPSK)
FFTSize = 48;            % FFT size for OFDM
k = log2(M);             % Bits per symbol (log base 2 of modulation order)
numSC = 48;              % Number of subcarriers
numBitSymbol = numSC * k; % Total number of bits per OFDM symbol
H = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;

MaxNumOfSymbols = 1000;
%% Neuronal Network
templateNet = load("BinaryClassifier/BinaryNet.mat").net;
% hyperameters
numEpochs = 30;
miniBatchSize = 1;
learningRate = 0.001;

%% Application
ber = zeros(1, length(SNR_dB)); % Preallocate BER results
networks = cell(length(SNR_dB),M);

% Preallocate containers to store metrics and corresponding symbol indices
allMetrics = cell(length(SNR_dB), M-1, 1);
symbolVec = zeros(length(SNR_dB), M-1, 1);

for i = 1:length(SNR_dB)
    for j = 1:M
        % Create and assign your dlnetwork object.
        % Replace "layerGraph" with your actual layer graph or construction method.
        networks{i, j} = templateNet;
    end
end

%% Main Loop for each SNR
for i = 1:length(SNR_dB)

    % 3D array for binary masks
    signalTx    = zeros(numSC, M, MaxNumOfSymbols);  
    signalToFix = zeros(numSC, MaxNumOfSymbols);
    labels      = zeros(numSC, MaxNumOfSymbols);
    
    for symbolsNumber = 1 : MaxNumOfSymbols
        % Generate random data symbols
        rawSymbols = generateRandomData(M, numSC);
        labels(:,symbolsNumber) = rawSymbols;

        % Create binary masks for each symbol value
        for symbol = 0:M-1
            signalTx(:, symbol + 1, symbolsNumber) = (rawSymbols ~= symbol);
        end

        % Transmit and receive the signal through the channel
        [label, signalEstimate] = processChannelAndTransmit(rawSymbols, M, FFTSize, SNR_dB(i), numSC,H);
        signalToFix(:,symbolsNumber) = (angle(signalEstimate)+pi)/(2*pi);
        vector = signalToFix(:,symbolsNumber);

        % Calcular la media y la desviación estándar del vector completo
        mean_x = mean(vector);

        % Inicializar la matriz de correlación
        corr_matrix = zeros(48, 48);

        % Calcular la correlación entre cada par de elementos correctamente
        for row = 1:48
            for col = 1:48
                corr_matrix(row, col) = (vector(row) - mean_x) * (vector(col) - mean_x);
            end
        end
        log_corr = sqrt(corr_matrix);

        % Crear la figura con un tamaño grande para evitar que los números se solapen
        figure('Position', [100, 100, 1400, 900]); 

        % Calcular logaritmo complejo de la matriz de correlación
        %log_corr = log(corr_matrix);

        % Subplot 1: Ángulo (fase) de log(corr_matrix)
        subplot(2,2,1);
        imagesc(angle(log_corr));  
        colorbar;
        colormap(jet);
        title('Ángulo de log(corr_matrix)');
        xticks(1:length(rawSymbols));  
        yticks(1:length(rawSymbols));
        xticklabels(rawSymbols);  
        yticklabels(rawSymbols);
        xtickangle(45);
        xlabel('Símbolos');
        ylabel('Símbolos');
        axis square;

        % Subplot 2: Magnitud de log(corr_matrix)
        subplot(2,2,2);
        imagesc(abs(log_corr));  
        colorbar;
        colormap(jet);
        title('Magnitud de log(corr_matrix)');
        xticks(1:length(rawSymbols));  
        yticks(1:length(rawSymbols));
        xticklabels(rawSymbols);  
        yticklabels(rawSymbols);
        xtickangle(45);
        xlabel('Símbolos');
        ylabel('Símbolos');
        axis square;

        % Subplot 3: Parte Real de log(corr_matrix)
        subplot(2,2,3);
        imagesc(real(log_corr));  
        colorbar;
        colormap(jet);
        title('Parte Real de log(corr_matrix)');
        xticks(1:length(rawSymbols));  
        yticks(1:length(rawSymbols));
        xticklabels(rawSymbols);  
        yticklabels(rawSymbols);
        xtickangle(45);
        xlabel('Símbolos');
        ylabel('Símbolos');
        axis square;

        % Subplot 4: Parte Imaginaria de log(corr_matrix)
        subplot(2,2,4);
        imagesc(imag(log_corr));  
        colorbar;
        colormap(jet);
        title('Parte Imaginaria de log(corr_matrix)');
        xticks(1:length(rawSymbols));  
        yticks(1:length(rawSymbols));
        xticklabels(rawSymbols);  
        yticklabels(rawSymbols);
        xtickangle(45);
        xlabel('Símbolos');
        ylabel('Símbolos');
        axis square;

        % Ajustar el espaciado de los subplots y agregar un título general
        sgtitle('Visualización de log(corr_matrix)');

        % Ajustar tamaño de fuente en los ejes
        set(gca, 'FontSize', 12);

        % Guardar la imagen con alta resolución
        print('CorrPlot_4Subplots.png', '-dpng', '-r300'); % Guardar con 300 DPI
    end
end