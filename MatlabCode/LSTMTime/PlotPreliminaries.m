%% Parámetros del sistema
EbNo_dB = 0:2:16;          % Rango de Eb/No (energía por bit a ruido en dB)
modulationOrder = 4;       % Orden de la modulación DPSK (QPSK = 4)
fftSize = 48;              % Tamaño de FFT en OFDM (número de subportadoras)
bitsPerSymbol = log2(modulationOrder); % Bits transmitidos por símbolo DPSK
numSubcarriers = fftSize;  % Número de subportadoras
bitsPerOFDMSymbol = numSubcarriers * bitsPerSymbol;

useOnlyAWGNChannel = false; % Indica uso de canal AWGN (solo ruido)

% Conversión Eb/No a SNR
if useOnlyAWGNChannel
    SNR_dB = EbNo_dB + 10*log10(bitsPerSymbol);
else
    SNR_dB = EbNo_dB + 10*log10(bitsPerSymbol) + 10*log10(117/128);
end

figure;
berTheoretical = berawgn(EbNo_dB, 'dpsk', modulationOrder);
berDPSKFD = [0.3327 0.2815 0.2278 0.1939 0.1594 0.1314 0.1122 0.0934 0.0908];
berLMMSEFD = [0.2846 0.1919 0.1157 0.0511 0.0161 0.0028 0.0005 0.0003 0.0002];

semilogy(EbNo_dB, berLMMSEFD, 'kd-', 'LineWidth', 1.5);
hold on;
semilogy(EbNo_dB, berDPSKFD, 'k*-', 'LineWidth', 1.5);

BER_NONN = load("BERNoNN.mat").bitErrorRate;
BERN_NN = load("BERNN.mat").bitErrorRate;
BER_ZERO = load("CNN_BiLSTM_V2V/BER_CNN_BiLSTM.mat").bitErrorRate;

semilogy(EbNo_dB, BER_NONN, 'bo--', 'LineWidth', 1.5);
semilogy(EbNo_dB, BERN_NN, '-*', 'LineWidth', 1.5);
semilogy(EbNo_dB, BER_ZERO, '--', 'LineWidth', 1.5);
xlabel('Eb/No (dB)');
ylabel('BER');
legend('DPSK-OFDM coherent FD, LMMSE', 'DPSK-OFDM non-coherent FD', 'DPSK-OFDM non-coherent TD', 'DPSK-OFDM NN TD', 'DPSK CNN + LSTM');
grid on;
title('DPSK-OFDM non-coherent');
saveas(gcf,'Results.png')