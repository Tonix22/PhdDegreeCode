function [signalTx, signalEstimate] = processChannelAndTransmit(H, M, FFTSize, SNR_dB, numSC)
    % Generate random data symbols
    signalTx = generateRandomData(M, numSC);
    
    % Modulate data with PSK
    pskSignal = applyPSKModulation(signalTx, M);

    % Differential encoding for DPSK
    DPSKsignalTx = applyDPSKEncoding(pskSignal);

    % Process channel
    G = processChannel(H);
    RxSignal = G * DPSKsignalTx;

    % OFDM modulation
    OFDMsignalTx = ofdmModulate(RxSignal, FFTSize);

    % Pass through AWGN channel using SNR
    signalRx = awgn(OFDMsignalTx, SNR_dB);

    % OFDM demodulation
    OFDMsignalRx = ofdmDemodulate(signalRx, FFTSize);

    % Differential decoding for DPSK
    DPSKsignalRx = applyDPSKDecoding(OFDMsignalRx, numSC);

    % PSK demodulation
    signalEstimate = applyPSKDemodulation(DPSKsignalRx, M);
end
