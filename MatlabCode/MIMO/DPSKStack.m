close all;
clc;
addpath('../Libraries');

%% Leer JSON desde argumento de ejecución
p = inputParser;
addRequired(p, 'jsonPath', @ischar);
parse(p, getenv('jsonPath'));

jsonPath = p.Results.jsonPath;

if exist(jsonPath, 'file') ~= 2
    error('El archivo JSON no existe: %s', jsonPath);
end

% Leer el archivo JSON
jsonText = fileread(jsonPath);
config = jsondecode(jsonText);

% Asignar parámetros desde JSON
EbNo = config.EbNo(:);  % Asegurar que sea un vector columna
M = config.M;                        % Modulation order (QPSK)
FFTSize = config.FFTSize;            % FFT size for OFDM
Retransmitions = config.Retransmissions;  % Number of retransmissions
k = log2(M);                         % Bits per symbol (log base 2 of modulation order)
numSC = config.numSC;                 % Number of subcarriers
numBitSymbol = numSC * k;             % Total number of bits per OFDM symbol
samplesPerSNR = config.samplesPerSNR; % Samples per SNR value
V2VChannel = config.V2VChannel;

% Cargar el canal desde el archivo MAT si está habilitado
if islogical(V2VChannel) && V2VChannel
    disp('USING CHANNEL')
    H = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;
    SNR_dB_Range = EbNo + 10*log10(numBitSymbol);
else
    SNR_dB_Range = EbNo + 10*log10(k);
    disp('NO CHANNEL')
end

% SOLUCIÓN: Iterar correctamente sobre cada elemento de SNR_dB_Range
for idx = 1:length(SNR_dB_Range)
    SNR_dB = SNR_dB_Range(idx);  % Tomar cada elemento individualmente
    disp(['Procesando SNR = ', num2str(SNR_dB)]);  % Confirmación visual

    mimoSignal = zeros(samplesPerSNR, Retransmitions + 1, FFTSize);
    Tx         = zeros(samplesPerSNR, numSC);

    for s = 1:samplesPerSNR
        signalTxBits = randi([0 1], numBitSymbol,1);
        signalTx = bit2int(signalTxBits, k); % Convert bits to symbols
        Tx(s, :) = signalTx; % Store the signal in the matrix
        
        %% Stack Generation procedure
        for i = 1:Retransmitions
            
            if islogical(V2VChannel) && V2VChannel
                [~, DPSKsignalRx] = processChannelAndTransmit(signalTx, M, FFTSize, SNR_dB, numSC, H);
            else
                [~, DPSKsignalRx] = processChannelAndTransmit(signalTx, M, FFTSize, SNR_dB, numSC);
            end

            mimoSignal(s, i, :) = DPSKsignalRx;
        end

        for i = 1:FFTSize
            % Extraer los símbolos complejos para todas las retransmisiones
            symbols = mimoSignal(s, 1:Retransmitions, i);  % Números complejos
            % Calcular la Media Circular (Ángulo Medio)
            mean_angle = angle(mean(exp(1j * angle(symbols))));  % Media de vectores unitarios
            mimoSignal(s, 1:Retransmitions, i) = angle(symbols);
            mimoSignal(s, end, i) = mean_angle;
            mimoSignal(s, :, i) = (mimoSignal(s, :, i) + pi) / (2 * pi);
        end
    end

    %% Guardar mimoSignal usando Python dentro de MATLAB
    filename_rx = sprintf("../../PythonCode/DeepLearning/MIMOSolution/data/Signal_EbNo_Rx_%d.mat", floor(EbNo(idx)));
    save(filename_rx, 'mimoSignal');
    disp(['Saved ' filename_rx ' successfully.']);

    filename_tx = sprintf("../../PythonCode/DeepLearning/MIMOSolution/data/Signal_EbNo_Tx_%d.mat", floor(EbNo(idx)));
    save(filename_tx, 'Tx');
    disp(['Saved ' filename_tx ' successfully.']);
end
