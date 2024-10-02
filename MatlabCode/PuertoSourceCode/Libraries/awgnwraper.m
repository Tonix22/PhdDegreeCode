function signalRx = awgnwraper(OFDMsignalTx, SNR_dB)
    signalRx = awgn(OFDMsignalTx, SNR_dB);
end

