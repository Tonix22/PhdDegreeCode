function signalEstimate = applyPSKDemodulation(DPSKsignalRx, M)
    % Normalize the phase differences to the range [0, M-1]
    Norm_Factor = M / (2 * pi);
    signalEstimate = mod(round(angle(DPSKsignalRx) * Norm_Factor), M);
end
