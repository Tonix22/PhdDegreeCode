function DPSKsignalRx = applyDPSKDecoding(OFDMsignalRx, numSC)
    % Unwrap the phase of the received OFDM signal
    phaseUnwrapped = unwrap(angle(OFDMsignalRx));
    
    % Calculate the phase difference
    phaseDifference = diff([0; phaseUnwrapped]); % Prepend 0 for the first symbol
    
    % Reconstruct the complex signal from the phase differences
    DPSKsignalRx = exp(1j * phaseDifference);
end
