clear all
close all
clc

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
load('../../../Data/kaggle_dataset/v2v80211p_LOS.mat', 'vectReal32b')
channelLOS = vectReal32b;

load('../../../Data/kaggle_dataset/v2v80211p_NLOS.mat', 'vectReal32b')
channelNLOS = vectReal32b;

%% Simulación principal sobre valores de SNR
for snrIdx = 1:length(SNR_dB)
    
    numErrorsAccumulated = 0;  
    totalBitsTransmitted = 0;  
    channelIndex = 1;           
    
    while numErrorsAccumulated < 1e3 && totalBitsTransmitted < 1e7
        
        % 1. Genera un frame completo de símbolos OFDM solo una vez por cada iteración del while
        transmittedBitsFrame = randi([0 1], fftSize, bitsPerOFDMSymbol);
        transmittedFrame = zeros(fftSize, fftSize);
        
        for carrierIdx = 1:fftSize
            txBits = transmittedBitsFrame(carrierIdx,:)';
            txSymbols = bit2int(txBits, bitsPerSymbol);
            transmittedFrame(carrierIdx,:) = dpskmod(txSymbols, modulationOrder, pi/4);
        end

        receivedFrame = zeros(fftSize, fftSize);

        % 2. Transmisión símbolo por símbolo a través del canal y recepción
        for symbolIndexInFrame = 1:fftSize
            
            % OFDM modulación del símbolo actual
            ofdmSignalTx = ifft(transmittedFrame(:, symbolIndexInFrame), fftSize);

            % Simulación del canal AWGN
            signalPower_dB = 10*log10(var(ofdmSignalTx));
            noiseVariance = 10.^(0.1*(signalPower_dB - SNR_dB(snrIdx)));
            noise = sqrt(noiseVariance/2)*(randn(size(ofdmSignalTx)) + 1j*randn(size(ofdmSignalTx)));
            receivedSignal = ofdmSignalTx + noise;

            % Demodulación OFDM
            ofdmSignalRx = fft(receivedSignal, fftSize);

            % Canal combinado LOS + NLOS
            combinedChannel = channelLOS(:,:,channelIndex) + channelNLOS(:,:,channelIndex);

            % Señal afectada por el canal
            receivedFrame(:, symbolIndexInFrame) = combinedChannel * ofdmSignalRx;
        end

        % 3. Demodulación DPSK de todo el frame recibido
        for subcarrierIdx = 1:fftSize
            demodulatedSymbols = dpskdemod(receivedFrame(subcarrierIdx,:), modulationOrder, pi/4);
            receivedBits = int2bit(demodulatedSymbols', bitsPerSymbol);
            
            % Contar errores por símbolo
            errorsInCurrentSymbol = biterr(transmittedBitsFrame(subcarrierIdx,:)', receivedBits);
            numErrorsAccumulated = numErrorsAccumulated + errorsInCurrentSymbol;
            totalBitsTransmitted = totalBitsTransmitted + bitsPerOFDMSymbol;
            
            % Incrementar canal
            channelIndex = mod(channelIndex, 9999) + 1;
        end

    end

    bitErrorRate(snrIdx) = numErrorsAccumulated / totalBitsTransmitted;

end


%% Gráfica de resultados
figure;

berTheorical = berawgn(EbNo_dB,'dpsk',modulationOrder);
berDPSKFD = [ 0.3327    0.2815    0.2278    0.1939    0.1594    0.1314    0.1122    0.0934    0.0908];
berLMMSEFD = [0.2846    0.1919    0.1157    0.0511    0.0161    0.0028    0.0005    0.0003    0.0002];%[ 0.2432    0.1573    0.0890    0.0322    0.0080    0.0013    0.0004]

semilogy(EbNo_dB,berLMMSEFD,'kd-', 'LineWidth', 1.5)
hold on 
semilogy(EbNo_dB,berDPSKFD,'k*-','LineWidth', 1.5)
semilogy(EbNo_dB,bitErrorRate,'bo--', 'LineWidth', 1.5)
xlabel('Eb/No (dB)')
ylabel('BER')
legend('DPSK-OFDM coherent FD, LMMSE','DPSK-OFDM non-coherent FD', 'DPSK-OFDM non-coherent TD ')
grid
title('DPSK-OFDM non-coherent')
