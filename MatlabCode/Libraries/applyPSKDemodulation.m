function signalEstimate = applyPSKDemodulation(DPSKsignalRx, M)
    signalEstimate = pskdemod(DPSKsignalRx, M, pi/2);
end
