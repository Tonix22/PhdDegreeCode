function [signalEstimate,DPSKsignalRx] = processChannelAndTransmit(signalTx, M, FFTSize, SNR_dB, numSC, varargin)

    % Modulate data with PSK
    pskSignal = applyDPSKModulation(signalTx, M);

    % OFDM modulation
    OFDMsignalTx = ofdmModulate(pskSignal, FFTSize);

    if length(varargin) == 1
        G = processChannel(varargin{1});  % G in frequency domain

        % Convert to frequency domain
        freqSignal = fft(OFDMsignalTx);

        % Apply channel per subcarrier
        freqSignal = G * freqSignal;

        % Back to time domain
        OFDMsignalTx = ifft(freqSignal);
    end

    % Pass through AWGN channel using SNR
    signalRx = awgn(OFDMsignalTx, SNR_dB, "measured");

    % OFDM demodulation
    OFDMsignalRx = ofdmDemodulate(signalRx, FFTSize);

    % Differential decoding for DPSK
    DPSKsignalRx = applyDPSKDecoding(OFDMsignalRx, numSC);

    % PSK demodulation
    signalEstimate = applyPSKDemodulation(DPSKsignalRx, M);

end
