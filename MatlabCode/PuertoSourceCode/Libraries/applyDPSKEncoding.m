function DPSKsignalTx = applyDPSKEncoding(pskSignal)
    DPSKsignalTx = pskSignal;
    for n = 2:length(DPSKsignalTx)
        DPSKsignalTx(n) = DPSKsignalTx(n) * DPSKsignalTx(n-1);
    end
end
