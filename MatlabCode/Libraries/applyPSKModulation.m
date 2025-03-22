function pskSignal = applyPSKModulation(signalTx, M)
    pskSignal = pskmod(signalTx, M, pi/2);
end
