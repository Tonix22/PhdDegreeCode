function [signalEstimate,DPSKsignalRx] = processChannelAndTransmit(signalTx, M, FFTSize, SNR_dB, numSC, varargin)    
    % Modulate data with PSK
    pskSignal = applyPSKModulation(signalTx, M);

    % Differential encoding for DPSK
    DPSKsignalTx = applyDPSKEncoding(pskSignal);

    if length(varargin) == 1
        G = processChannel(varargin{1});  % Assign channel
        DPSKsignalTx = G * DPSKsignalTx;
    end

    % OFDM modulation
    OFDMsignalTx = ofdmModulate(DPSKsignalTx, FFTSize);

    % Pass through AWGN channel using SNR
    signalRx = awgn(OFDMsignalTx, SNR_dB);

    % OFDM demodulation
    OFDMsignalRx = ofdmDemodulate(signalRx, FFTSize);

    % Differential decoding for DPSK
    DPSKsignalRx = applyDPSKDecoding(OFDMsignalRx, numSC);

    % PSK demodulation
    signalEstimate = applyPSKDemodulation(DPSKsignalRx, M);
end
