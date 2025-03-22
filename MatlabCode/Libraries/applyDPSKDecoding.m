function DPSKsignalRx = applyDPSKDecoding(OFDMsignalRx, numSC)
    DPSKsignalRx = complex(zeros(numSC, 1));
    DPSKsignalRx(1) = OFDMsignalRx(1); % First symbol remains unchanged
    DPSKsignalRx(2:end) = OFDMsignalRx(2:end) .* conj(OFDMsignalRx(1:end-1));
end
