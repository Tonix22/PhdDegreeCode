close all
clear all
clc
addpath('../Libraries');

%% parameter system
EbNo = 0:2:12;
M = 4; % Modulation order
FFTSize = 64;
k = log2(M);
numSC = FFTSize;
numBitSymbol = numSC * k;

channelAWGN = 1;

if channelAWGN
    SNR = EbNo + 10*log10(k);
else
    SNR = EbNo + 10*log10(numBitSymbol);
end

ber = zeros(1,length(EbNo));

for i = 1:length(SNR)
    numError = 0;
    numBits = 0;

    while numError < 1000
    
        %Generate random data symbols
        %signalTx = randi([0 1], numBitSymbol,1);
        signalTxBits2 = repmat([0; 1], numBitSymbol / 2, 1); % Alterna entre 0 y 1
        symbolTx2 = bit2int(signalTxBits,k);
        
        %QAM modulate 
        
        pskSignal2 = dpskmod(symbolTx,M);
       
        %OFDM modulate
        OFDMsignalTx2 = ifft(pskSignal2,FFTSize);
        
        %pass throuht channel
        
        signalRx = awgn(OFDMsignalTx,SNR(i),"measured");

        %OFDM demodulate
        OFDMsignalRx = fft(signalRx,FFTSize);

        %*****************************************************
        % Differential decoding for DPSK
        DPSKsignalRx = applyDPSKDecoding(OFDMsignalRx, numSC);

        % PSK demodulation
        signalEstimateMyMethod = applyPSKDemodulation(DPSKsignalRx, M);
        %*****************************************************

        bitsRx = int2bit(signalEstimateMyMethod,k);
        
        numErrorCalculate = biterr(signalTx,bitsRx);
        
        
        numError = numError + numErrorCalculate;
        numBits = numBits + numBitSymbol;


    end

ber(i) = numError/numBits

end


% filepath: /home/tonix/Documents/PhdDegreeCode/MatlabCode/DPSK/OFDM_DPKS.m
berTheorical = berawgn(EbNo,'dpsk',M);

semilogy(EbNo,berTheorical,'b-', 'LineWidth', 1.5)
hold on 
semilogy(EbNo,ber,'ko--', 'LineWidth', 1.5)
xlabel('Eb/No (dB)')
ylabel('BER')
legend('Theoretical BER', 'Estimate BER')
grid

% Save the figure as a .png file
outputFileName = 'BER_vs_EbNo.png';
print(outputFileName, '-dpng', '-r300'); % Save as PNG with 300 DPI resolution