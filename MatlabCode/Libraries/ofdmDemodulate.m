function OFDMsignalRx = ofdmDemodulate(signalRx, FFTSize)
    OFDMsignalRx = fft(signalRx, FFTSize);
end
