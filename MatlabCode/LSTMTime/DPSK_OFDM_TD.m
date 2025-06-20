clear all
close all
clc
addpath('Helper');
%% Parámetros del sistema
EbNo_dB = 0:2:16;          % Rango de Eb/No (energía por bit a ruido en dB)
modulationOrder = 4;       % Orden de la modulación DPSK (QPSK = 4)
fftSize = 48;              % Tamaño de FFT en OFDM (número de subportadoras)
bitsPerSymbol = log2(modulationOrder); % Bits transmitidos por símbolo DPSK
numSubcarriers = fftSize;  % Número de subportadoras
bitsPerOFDMSymbol = numSubcarriers * bitsPerSymbol;

useOnlyAWGNChannel = true; % Indica uso de canal AWGN (solo ruido)

% Conversión Eb/No a SNR
if useOnlyAWGNChannel
    SNR_dB = EbNo_dB + 10*log10(bitsPerSymbol);
else
    SNR_dB = EbNo_dB + 10*log10(bitsPerSymbol) + 10*log10(117/128);
end

bitErrorRate = zeros(1, length(EbNo_dB)); % Inicialización de vector BER

%% Carga de datos del canal V2V (LOS y NLOS)
load('../../Data/kaggle_dataset/v2v80211p_LOS.mat', 'vectReal32b')
channelLOS = vectReal32b;

load('../../Data/kaggle_dataset/v2v80211p_NLOS.mat', 'vectReal32b')
channelNLOS = vectReal32b;

%% Simulación principal sobre valores de SNR
for snrIdx = 1:length(SNR_dB)
    
    numErrorsAccumulated = 0;  
    totalBitsTransmitted = 0;  
    channelIndex = 1;           
    
    while numErrorsAccumulated < 1e3 && totalBitsTransmitted < 1e7
        
        % 1. Genera un frame completo de símbolos OFDM solo una vez por cada iteración del while
        [transmittedBitsFrame, ~, transmittedFrame] = generateTransmittedFrame(fftSize, bitsPerOFDMSymbol, bitsPerSymbol, modulationOrder);

        % 2. Transmisión símbolo por símbolo a través del canal y recepción
        receivedFrame = transmitThroughChannel(transmittedFrame, fftSize, SNR_dB(snrIdx), channelLOS, channelNLOS, channelIndex);

        % 3. Demodulación DPSK de todo el frame recibido
        [errFrame, ~, channelIndex] = decodeAndCountErrors(receivedFrame, transmittedBitsFrame, ...
        modulationOrder, bitsPerSymbol, channelIndex);

        numErrorsAccumulated  = numErrorsAccumulated  + errFrame;
        totalBitsTransmitted  = totalBitsTransmitted  + (fftSize * bitsPerOFDMSymbol);
    end

    bitErrorRate(snrIdx) = numErrorsAccumulated / totalBitsTransmitted;

end

save('BERNoNN.mat','bitErrorRate');
%% Gráfica de resultados
plotBERResults(EbNo_dB,bitErrorRate,modulationOrder);